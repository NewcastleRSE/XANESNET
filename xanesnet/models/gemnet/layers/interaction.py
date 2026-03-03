"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch

from .atom_update import AtomUpdateBlock
from .base import Dense, ResidualLayer
from .efficient import EfficientInteractionBilinear
from .embedding import EdgeEmbedding
from .scaling import ScalingFactor


class InteractionBlock(torch.nn.Module):
    """
    Interaction block for GemNet-Q/dQ.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_quad: int
            (Down-projected) Embedding size in the quadruplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        emb_size_bil_quad: int
            Embedding size of the edge embeddings in the quadruplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_quad: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        emb_size_bil_trip: int,
        emb_size_bil_quad: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        activation: str,
        scale_file: str | None,
        name: str = "Interaction",
    ) -> None:
        super().__init__()
        self.name = name

        block_nr = name.split("_")[-1]

        ## -------------------------------------------- Message Passing ------------------------------------------- ##
        # Dense transformation of skip connection
        self.dense_ca = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Quadruplet Interaction
        self.quad_interaction = QuadrupletInteraction(
            emb_size_edge=emb_size_edge,
            emb_size_quad=emb_size_quad,
            emb_size_bilinear=emb_size_bil_quad,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            emb_size_sbf=emb_size_sbf,
            activation=activation,
            scale_file=scale_file,
            name=f"QuadInteraction_{block_nr}",
        )

        # Triplet Interaction
        self.trip_interaction = TripletInteraction(
            emb_size_edge=emb_size_edge,
            emb_size_trip=emb_size_trip,
            emb_size_bilinear=emb_size_bil_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            activation=activation,
            scale_file=scale_file,
            name=f"TripInteraction_{block_nr}",
        )

        ## ---------------------------------------- Update Edge Embeddings ---------------------------------------- ##
        # Residual layers before skip connection
        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_before_skip)]
        )

        # Residual layers after skip connection
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_after_skip)]
        )

        ## ---------------------------------------- Update Atom Embeddings ---------------------------------------- ##
        self.atom_update = AtomUpdateBlock(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=num_atom,
            activation=activation,
            scale_file=scale_file,
            name=f"AtomUpdate_{block_nr}",
        )

        ## ------------------------------ Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        self.concat_layer = EdgeEmbedding(
            emb_size_atom,
            emb_size_edge,
            emb_size_edge,
            activation=activation,
        )
        self.residual_m = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_concat)]
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)
        self.inv_sqrt_3 = 1 / (3.0**0.5)

    def reset_parameters(self) -> None:
        self.dense_ca.reset_parameters()
        self.quad_interaction.reset_parameters()
        self.trip_interaction.reset_parameters()
        for layer in self.layers_before_skip:
            layer.reset_parameters()
        for layer in self.layers_after_skip:
            layer.reset_parameters()
        self.atom_update.reset_parameters()
        self.concat_layer.reset_parameters()
        for layer in self.residual_m:
            layer.reset_parameters()

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        rbf4: torch.Tensor,
        cbf4: torch.Tensor,
        sbf4: tuple[torch.Tensor, torch.Tensor],
        Kidx4: torch.Tensor,
        rbf3: torch.Tensor,
        cbf3: tuple[torch.Tensor, torch.Tensor],
        Kidx3: torch.Tensor,
        id_swap: torch.Tensor,
        id3_expand_ba: torch.Tensor,
        id3_reduce_ca: torch.Tensor,
        id4_reduce_ca: torch.Tensor,
        id4_expand_intm_db: torch.Tensor,
        id4_expand_abd: torch.Tensor,
        rbf_h: torch.Tensor,
        id_c: torch.Tensor,
        id_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
            h: Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        # Initial transformation
        x_ca_skip = self.dense_ca(m)  # (nEdges, emb_size_edge)

        x4 = self.quad_interaction(
            m,
            rbf4,
            cbf4,
            sbf4,
            Kidx4,
            id_swap,
            id4_reduce_ca,
            id4_expand_intm_db,
            id4_expand_abd,
        )
        x3 = self.trip_interaction(m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca)

        ## ---------------------- Merge Embeddings after Quadruplet and Triplet Interaction ---------------------- ##
        x = x_ca_skip + x3 + x4  # (nEdges, emb_size_edge)
        x = x * self.inv_sqrt_3

        ## --------------------------------------- Update Edge Embeddings ---------------------------------------- ##
        # Transformations before skip connection
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + x  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2

        # Transformations after skip connection
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)  # (nEdges, emb_size_edge)

        ## --------------------------------------- Update Atom Embeddings ---------------------------------------- ##
        h2 = self.atom_update(h, m, rbf_h, id_a)

        # Skip connection
        h = h + h2  # (nAtoms, emb_size_atom)
        h = h * self.inv_sqrt_2

        ## ----------------------------- Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        m2 = self.concat_layer(h, m, id_c, id_a)  # (nEdges, emb_size_edge)

        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + m2  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2
        return h, m


class InteractionBlockTripletsOnly(torch.nn.Module):
    """
    Interaction block for GemNet-T/dT.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        activation: str,
        scale_file: str | None,
        name: str = "Interaction",
        # Accept and ignore extra kwargs for compatibility with InteractionBlock
        # (allows the same constructor call for both GemNet-T and GemNet-Q variants)
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.name = name

        block_nr = name.split("_")[-1]

        ## -------------------------------------------- Message Passing ------------------------------------------- ##
        # Dense transformation of skip connection
        self.dense_ca = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Triplet Interaction
        self.trip_interaction = TripletInteraction(
            emb_size_edge=emb_size_edge,
            emb_size_trip=emb_size_trip,
            emb_size_bilinear=emb_size_bil_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            activation=activation,
            scale_file=scale_file,
            name=f"TripInteraction_{block_nr}",
        )

        ## ---------------------------------------- Update Edge Embeddings ---------------------------------------- ##
        # Residual layers before skip connection
        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_before_skip)]
        )

        # Residual layers after skip connection
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_after_skip)]
        )

        ## ---------------------------------------- Update Atom Embeddings ---------------------------------------- ##
        self.atom_update = AtomUpdateBlock(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=num_atom,
            activation=activation,
            scale_file=scale_file,
            name=f"AtomUpdate_{block_nr}",
        )

        ## ------------------------------ Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        self.concat_layer = EdgeEmbedding(
            emb_size_atom,
            emb_size_edge,
            emb_size_edge,
            activation=activation,
        )
        self.residual_m = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_concat)]
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)

    def reset_parameters(self) -> None:
        self.dense_ca.reset_parameters()
        self.trip_interaction.reset_parameters()
        for layer in self.layers_before_skip:
            layer.reset_parameters()
        for layer in self.layers_after_skip:
            layer.reset_parameters()
        self.atom_update.reset_parameters()
        self.concat_layer.reset_parameters()
        for layer in self.residual_m:
            layer.reset_parameters()

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        rbf3: torch.Tensor,
        cbf3: tuple[torch.Tensor, torch.Tensor],
        Kidx3: torch.Tensor,
        id_swap: torch.Tensor,
        id3_expand_ba: torch.Tensor,
        id3_reduce_ca: torch.Tensor,
        rbf_h: torch.Tensor,
        id_c: torch.Tensor,
        id_a: torch.Tensor,
        **kwargs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
            h: Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        # Initial transformation
        x_ca_skip = self.dense_ca(m)  # (nEdges, emb_size_edge)

        x3 = self.trip_interaction(m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca)

        ## ----------------------------- Merge Embeddings after Triplet Interaction ------------------------------ ##
        x = x_ca_skip + x3  # (nEdges, emb_size_edge)
        x = x * self.inv_sqrt_2

        ## ---------------------------------------- Update Edge Embeddings --------------------------------------- ##
        # Transformations before skip connection
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + x  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2

        # Transformations after skip connection
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)  # (nEdges, emb_size_edge)

        ## ---------------------------------------- Update Atom Embeddings --------------------------------------- ##
        h2 = self.atom_update(h, m, rbf_h, id_a)  # (nAtoms, emb_size_atom)

        # Skip connection
        h = h + h2  # (nAtoms, emb_size_atom)
        h = h * self.inv_sqrt_2

        ## ----------------------------- Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        m2 = self.concat_layer(h, m, id_c, id_a)  # (nEdges, emb_size_edge)

        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + m2  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2
        return h, m


class QuadrupletInteraction(torch.nn.Module):
    """
    Quadruplet-based message passing block.

    Parameters
    ----------
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_quad: int
            (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear: int
            Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_edge: int,
        emb_size_quad: int,
        emb_size_bilinear: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        activation: str,
        scale_file: str | None,
        name: str = "QuadrupletInteraction",
    ) -> None:
        super().__init__()
        self.name = name

        # Dense transformation
        self.dense_db = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Up projections of basis representations, bilinear layer and scaling factors
        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had_rbf")

        self.mlp_cbf = Dense(emb_size_cbf, emb_size_quad, activation=None, bias=False)
        self.scale_cbf = ScalingFactor(scale_file=scale_file, name=name + "_had_cbf")

        self.mlp_sbf = EfficientInteractionBilinear(emb_size_quad, emb_size_sbf, emb_size_bilinear)
        self.scale_sbf_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum_sbf"
        )  # combines scaling for bilinear layer and summation

        # Down and up projections
        self.down_projection = Dense(
            emb_size_edge,
            emb_size_quad,
            activation=activation,
            bias=False,
        )
        self.up_projection_ca = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )
        self.up_projection_ac = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)

    def reset_parameters(self) -> None:
        self.dense_db.reset_parameters()
        self.mlp_rbf.reset_parameters()
        self.mlp_cbf.reset_parameters()
        self.mlp_sbf.reset_parameters()
        self.down_projection.reset_parameters()
        self.up_projection_ca.reset_parameters()
        self.up_projection_ac.reset_parameters()

    def forward(
        self,
        m: torch.Tensor,
        rbf: torch.Tensor,
        cbf: torch.Tensor,
        sbf: tuple[torch.Tensor, torch.Tensor],
        Kidx4: torch.Tensor,
        id_swap: torch.Tensor,
        id4_reduce_ca: torch.Tensor,
        id4_expand_intm_db: torch.Tensor,
        id4_expand_abd: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_db = self.dense_db(m)  # (nEdges, emb_size_edge)

        # Transform via radial bessel basis
        x_db2 = x_db * self.mlp_rbf(rbf)  # (nEdges, emb_size_edge)
        x_db = self.scale_rbf(x_db, x_db2)

        # Down project embeddings
        x_db = self.down_projection(x_db)  # (nEdges, emb_size_quad)

        # Transform via circular spherical bessel basis
        x_db = x_db[id4_expand_intm_db]  # (intmTriplets, emb_size_quad)
        x_db2 = x_db * self.mlp_cbf(cbf)  # (intmTriplets, emb_size_quad)
        x_db = self.scale_cbf(x_db, x_db2)

        # Transform via spherical bessel basis
        x_db = x_db[id4_expand_abd]  # (nQuadruplets, emb_size_quad)
        x = self.mlp_sbf(sbf, x_db, id4_reduce_ca, Kidx4)  # (nEdges, emb_size_bilinear)
        x = self.scale_sbf_sum(x_db, x)

        # Basis representation:
        # rbf(d_db)
        # cbf(d_ba, angle_abd)
        # sbf(d_ca, angle_cab, angle_cabd)

        # Upproject embeddings
        x_ca = self.up_projection_ca(x)  # (nEdges, emb_size_edge)
        x_ac = self.up_projection_ac(x)  # (nEdges, emb_size_edge)

        # Merge interaction of c->a and a->c
        x_ac = x_ac[id_swap]  # swap to add to edge a->c and not c->a
        x4 = x_ca + x_ac
        x4 = x4 * self.inv_sqrt_2

        return x4


class TripletInteraction(torch.nn.Module):
    """
    Triplet-based message passing block.

    Parameters
    ----------
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear: int
            Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_bilinear: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        activation: str,
        scale_file: str | None,
        name: str = "TripletInteraction",
    ) -> None:
        super().__init__()
        self.name = name

        # Dense transformation
        self.dense_ba = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Down projections of basis representations, bilinear layer and scaling factors
        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had_rbf")

        self.mlp_cbf = EfficientInteractionBilinear(emb_size_trip, emb_size_cbf, emb_size_bilinear)
        self.scale_cbf_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum_cbf"
        )  # combines scaling for bilinear layer and summation

        # Down and up projections
        self.down_projection = Dense(
            emb_size_edge,
            emb_size_trip,
            activation=activation,
            bias=False,
        )
        self.up_projection_ca = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )
        self.up_projection_ac = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        self.inv_sqrt_2 = 1 / (2.0) ** 0.5

    def reset_parameters(self) -> None:
        self.dense_ba.reset_parameters()
        self.mlp_rbf.reset_parameters()
        self.mlp_cbf.reset_parameters()
        self.down_projection.reset_parameters()
        self.up_projection_ca.reset_parameters()
        self.up_projection_ac.reset_parameters()

    def forward(
        self,
        m: torch.Tensor,
        rbf3: torch.Tensor,
        cbf3: tuple[torch.Tensor, torch.Tensor],
        Kidx3: torch.Tensor,
        id_swap: torch.Tensor,
        id3_expand_ba: torch.Tensor,
        id3_reduce_ca: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        # Dense transformation
        x_ba = self.dense_ba(m)  # (nEdges, emb_size_edge)

        # Transform via radial bessel basis
        mlp_rbf = self.mlp_rbf(rbf3)  # (nEdges, emb_size_edge)
        x_ba2 = x_ba * mlp_rbf
        x_ba = self.scale_rbf(x_ba, x_ba2)

        x_ba = self.down_projection(x_ba)  # (nEdges, emb_size_trip)

        # Transform via circular spherical basis
        x_ba = x_ba[id3_expand_ba]  # (nTriplets, emb_size_trip)

        # Efficient bilinear layer
        x = self.mlp_cbf(cbf3, x_ba, id3_reduce_ca, Kidx3)  # (nEdges, emb_size_bilinear)
        x = self.scale_cbf_sum(x_ba, x)

        # Basis representation:
        # rbf(d_ba)
        # cbf(d_ca, angle_cab)

        # Up project embeddings
        x_ca = self.up_projection_ca(x)  # (nEdges, emb_size_edge)
        x_ac = self.up_projection_ac(x)  # (nEdges, emb_size_edge)

        # Merge interaction of c->a and a->c
        x_ac = x_ac[id_swap]  # swap to add to edge a->c and not c->a
        x3 = x_ca + x_ac
        x3 = x3 * self.inv_sqrt_2
        return x3
