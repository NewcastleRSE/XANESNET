# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

"""Interaction blocks for GemNet-OC (triplet, quadruplet, atom-edge, and pair interactions)."""

import math

import torch

from .atom_update_block import AtomUpdateBlock
from .base_layers import Dense, ResidualLayer
from .efficient import EfficientInteractionBilinear
from .embedding_block import EdgeEmbedding
from .scaling import ScaleFactor


class InteractionBlock(torch.nn.Module):
    """Full GemNet-OC interaction block combining triplet, quadruplet, and pair interactions.

    Applies edge-to-edge (E2E), atom-to-edge (A2E), edge-to-atom (E2A), and
    atom-to-atom (A2A) message-passing paths as configured, then updates both
    the edge embeddings ``m`` and atom embeddings ``h``.

    Args:
        emb_size_atom: Atom embedding dimension.
        emb_size_edge: Edge embedding dimension.
        emb_size_trip_in: Input triplet embedding dimension.
        emb_size_trip_out: Output triplet embedding dimension.
        emb_size_quad_in: Input quadruplet embedding dimension.
        emb_size_quad_out: Output quadruplet embedding dimension.
        emb_size_a2a_in: Input atom-pair embedding dimension.
        emb_size_a2a_out: Output atom-pair embedding dimension.
        emb_size_rbf: Radial basis function embedding dimension.
        emb_size_cbf: Circular basis function embedding dimension.
        emb_size_sbf: Spherical basis function embedding dimension.
        num_before_skip: Number of residual layers before the skip connection.
        num_after_skip: Number of residual layers after the skip connection.
        num_concat: Number of residual layers after the concat layer.
        num_atom: Number of residual layers in the atom update block.
        num_atom_emb_layers: Extra atom residual layers before the update.
        quad_interaction: Enable quadruplet interactions.
        atom_edge_interaction: Enable atom-to-edge interactions.
        edge_atom_interaction: Enable edge-to-atom interactions.
        atom_interaction: Enable atom-to-atom pair interactions.
        activation: Activation function name.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip_in: int,
        emb_size_trip_out: int,
        emb_size_quad_in: int,
        emb_size_quad_out: int,
        emb_size_a2a_in: int,
        emb_size_a2a_out: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        num_atom_emb_layers: int = 0,
        quad_interaction: bool = False,
        atom_edge_interaction: bool = False,
        edge_atom_interaction: bool = False,
        atom_interaction: bool = False,
        activation: str | None = None,
    ) -> None:
        super().__init__()

        self.dense_ca = Dense(emb_size_edge, emb_size_edge, activation=activation, bias=False)

        self.trip_interaction = TripletInteraction(
            emb_size_in=emb_size_edge,
            emb_size_out=emb_size_edge,
            emb_size_trip_in=emb_size_trip_in,
            emb_size_trip_out=emb_size_trip_out,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            symmetric_mp=True,
            swap_output=True,
            activation=activation,
        )

        self.quad_interaction: QuadrupletInteraction | None
        if quad_interaction:
            self.quad_interaction = QuadrupletInteraction(
                emb_size_edge=emb_size_edge,
                emb_size_quad_in=emb_size_quad_in,
                emb_size_quad_out=emb_size_quad_out,
                emb_size_rbf=emb_size_rbf,
                emb_size_cbf=emb_size_cbf,
                emb_size_sbf=emb_size_sbf,
                symmetric_mp=True,
                activation=activation,
            )
        else:
            self.quad_interaction = None

        self.atom_edge_interaction: TripletInteraction | None
        if atom_edge_interaction:
            self.atom_edge_interaction = TripletInteraction(
                emb_size_in=emb_size_atom,
                emb_size_out=emb_size_edge,
                emb_size_trip_in=emb_size_trip_in,
                emb_size_trip_out=emb_size_trip_out,
                emb_size_rbf=emb_size_rbf,
                emb_size_cbf=emb_size_cbf,
                symmetric_mp=True,
                swap_output=True,
                activation=activation,
            )
        else:
            self.atom_edge_interaction = None
        self.edge_atom_interaction: TripletInteraction | None
        if edge_atom_interaction:
            self.edge_atom_interaction = TripletInteraction(
                emb_size_in=emb_size_edge,
                emb_size_out=emb_size_atom,
                emb_size_trip_in=emb_size_trip_in,
                emb_size_trip_out=emb_size_trip_out,
                emb_size_rbf=emb_size_rbf,
                emb_size_cbf=emb_size_cbf,
                symmetric_mp=False,
                swap_output=False,
                activation=activation,
            )
        else:
            self.edge_atom_interaction = None
        self.atom_interaction: PairInteraction | None
        if atom_interaction:
            self.atom_interaction = PairInteraction(
                emb_size_atom=emb_size_atom,
                emb_size_pair_in=emb_size_a2a_in,
                emb_size_pair_out=emb_size_a2a_out,
                emb_size_rbf=emb_size_rbf,
                activation=activation,
            )
        else:
            self.atom_interaction = None

        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_before_skip)]
        )
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_after_skip)]
        )
        self.atom_emb_layers = torch.nn.ModuleList(
            [ResidualLayer(emb_size_atom, activation=activation) for _ in range(num_atom_emb_layers)]
        )

        self.atom_update = AtomUpdateBlock(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=num_atom,
            activation=activation,
        )

        self.concat_layer = EdgeEmbedding(emb_size_atom, emb_size_edge, emb_size_edge, activation=activation)
        self.residual_m = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_concat)]
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        num_eint = 2.0 + quad_interaction + atom_edge_interaction
        self.inv_sqrt_num_eint = 1 / math.sqrt(num_eint)
        num_aint = 1.0 + edge_atom_interaction + atom_interaction
        self.inv_sqrt_num_aint = 1 / math.sqrt(num_aint)

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        bases_qint: dict[str, torch.Tensor],
        bases_e2e: dict[str, torch.Tensor],
        bases_a2e: dict[str, torch.Tensor],
        bases_e2a: dict[str, torch.Tensor],
        basis_a2a_rad: torch.Tensor | None,
        basis_atom_update: torch.Tensor,
        edge_index_main: torch.Tensor,
        a2ee2a_graph: dict[str, torch.Tensor],
        a2a_graph: dict[str, torch.Tensor],
        id_swap: torch.Tensor,
        trip_idx_e2e: dict[str, torch.Tensor],
        trip_idx_a2e: dict[str, torch.Tensor],
        trip_idx_e2a: dict[str, torch.Tensor],
        quad_idx: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one interaction block over atom and edge embeddings.

        Args:
            h: Atom embeddings, shape ``(nAtoms, emb_size_atom)``.
            m: Edge embeddings, shape ``(nEdges, emb_size_edge)``.
            bases_qint: Quadruplet interaction bases.
            bases_e2e: Edge-to-edge triplet bases.
            bases_a2e: Atom-to-edge triplet bases.
            bases_e2a: Edge-to-atom triplet bases.
            basis_a2a_rad: Atom-pair radial basis, or ``None``.
            basis_atom_update: Radial basis for the atom update block.
            edge_index_main: Main edge index, shape ``(2, nEdges)``.
            a2ee2a_graph: Atom-to-edge-to-atom graph dictionary.
            a2a_graph: Atom-to-atom graph dictionary.
            id_swap: Index mapping each edge to its reverse, shape
                ``(nEdges,)``.
            trip_idx_e2e: Triplet indices for E2E interactions.
            trip_idx_a2e: Triplet indices for A2E interactions.
            trip_idx_e2a: Triplet indices for E2A interactions.
            quad_idx: Quadruplet indices (nested dict).

        Returns:
            Updated ``(h, m)`` tuple of atom and edge embeddings.
        """
        num_atoms = h.shape[0]

        x_ca_skip = self.dense_ca(m)

        x_qint = x_a2e = h_e2a = h_a2a = None

        x_e2e = self.trip_interaction(m, bases_e2e, trip_idx_e2e, id_swap)
        if self.quad_interaction is not None:
            x_qint = self.quad_interaction(m, bases_qint, quad_idx, id_swap)
        if self.atom_edge_interaction is not None:
            x_a2e = self.atom_edge_interaction(
                h,
                bases_a2e,
                trip_idx_a2e,
                id_swap,
                expand_idx=a2ee2a_graph["edge_index"][0],
            )
        if self.edge_atom_interaction is not None:
            h_e2a = self.edge_atom_interaction(
                m,
                bases_e2a,
                trip_idx_e2a,
                id_swap,
                idx_agg2=a2ee2a_graph["edge_index"][1],
                idx_agg2_inner=a2ee2a_graph["target_neighbor_idx"],
                agg2_out_size=num_atoms,
            )
        if self.atom_interaction is not None:
            h_a2a = self.atom_interaction(
                h,
                basis_a2a_rad,
                a2a_graph["edge_index"],
                a2a_graph["target_neighbor_idx"],
            )

        x = x_ca_skip + x_e2e
        if self.quad_interaction is not None:
            x = x + x_qint
        if self.atom_edge_interaction is not None:
            x = x + x_a2e
        x = x * self.inv_sqrt_num_eint

        if self.edge_atom_interaction is not None:
            h = h + h_e2a
        if self.atom_interaction is not None:
            h = h + h_a2a
        h = h * self.inv_sqrt_num_aint

        for layer in self.layers_before_skip:
            x = layer(x)

        m = m + x
        m = m * self.inv_sqrt_2

        for layer in self.layers_after_skip:
            m = layer(m)

        for layer in self.atom_emb_layers:
            h = layer(h)

        h2 = self.atom_update(h, m, basis_atom_update, edge_index_main[1])
        h = h + h2
        h = h * self.inv_sqrt_2

        m2 = self.concat_layer(h, m, edge_index_main)
        for layer in self.residual_m:
            m2 = layer(m2)

        m = m + m2
        m = m * self.inv_sqrt_2
        return h, m


class QuadrupletInteraction(torch.nn.Module):
    """Quadruplet interaction using radial, circular, and spherical bases.

    Args:
        emb_size_edge: Edge embedding dimension.
        emb_size_quad_in: Input quadruplet embedding dimension.
        emb_size_quad_out: Output quadruplet embedding dimension.
        emb_size_rbf: Radial basis embedding dimension.
        emb_size_cbf: Circular basis embedding dimension.
        emb_size_sbf: Spherical basis embedding dimension.
        symmetric_mp: If ``True``, apply symmetric message passing (ca + ac
            scaled by ``1 / sqrt(2)``).
        activation: Activation function name.
    """

    def __init__(
        self,
        emb_size_edge: int,
        emb_size_quad_in: int,
        emb_size_quad_out: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        symmetric_mp: bool = True,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        self.symmetric_mp = symmetric_mp

        self.dense_db = Dense(emb_size_edge, emb_size_edge, activation=activation, bias=False)

        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_rbf = ScaleFactor()

        self.mlp_cbf = Dense(emb_size_cbf, emb_size_quad_in, activation=None, bias=False)
        self.scale_cbf = ScaleFactor()

        self.mlp_sbf = EfficientInteractionBilinear(emb_size_quad_in, emb_size_sbf, emb_size_quad_out)
        self.scale_sbf_sum = ScaleFactor()

        self.down_projection = Dense(emb_size_edge, emb_size_quad_in, activation=activation, bias=False)
        self.up_projection_ca = Dense(emb_size_quad_out, emb_size_edge, activation=activation, bias=False)
        if self.symmetric_mp:
            self.up_projection_ac = Dense(emb_size_quad_out, emb_size_edge, activation=activation, bias=False)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        m: torch.Tensor,
        bases: dict[str, torch.Tensor],
        idx: dict,
        id_swap: torch.Tensor,
    ) -> torch.Tensor:
        """Apply quadruplet interaction to edge embeddings.

        Args:
            m: Edge embeddings, shape ``(nEdges, emb_size_edge)``.
            bases: Dictionary with keys ``"rad"``, ``"cir"``, ``"sph"``
                holding the respective pre-computed bases.
            idx: Quadruplet index dictionary.
            id_swap: Edge-swap index, shape ``(nEdges,)``.

        Returns:
            Updated edge embeddings of shape
            ``(nEdges, emb_size_edge)``.
        """
        x_db = self.dense_db(m)

        x_db2 = x_db * self.mlp_rbf(bases["rad"])
        x_db = self.scale_rbf(x_db2, ref=x_db)

        x_db = self.down_projection(x_db)

        x_db = x_db[idx["triplet_in"]["in"]]
        x_db2 = x_db * self.mlp_cbf(bases["cir"])
        x_db = self.scale_cbf(x_db2, ref=x_db)

        x_db = x_db[idx["trip_in_to_quad"]]
        x = self.mlp_sbf(bases["sph"], x_db, idx["out"], idx["out_agg"])
        x = self.scale_sbf_sum(x, ref=x_db)

        if self.symmetric_mp:
            x_ca = self.up_projection_ca(x)
            x_ac = self.up_projection_ac(x)
            x_ac = x_ac[id_swap]
            x_res = x_ca + x_ac
            return x_res * self.inv_sqrt_2
        else:
            return self.up_projection_ca(x)


class TripletInteraction(torch.nn.Module):
    """Triplet interaction using radial and circular bases.

    Args:
        emb_size_in: Input embedding dimension (atom or edge).
        emb_size_out: Output embedding dimension (atom or edge).
        emb_size_trip_in: Intermediate triplet input embedding dimension.
        emb_size_trip_out: Intermediate triplet output embedding dimension.
        emb_size_rbf: Radial basis embedding dimension.
        emb_size_cbf: Circular basis embedding dimension.
        symmetric_mp: If ``True``, apply symmetric message passing.
        swap_output: If ``True``, apply ``id_swap`` when ``symmetric_mp``
            is ``False``.
        activation: Activation function name.
    """

    def __init__(
        self,
        emb_size_in: int,
        emb_size_out: int,
        emb_size_trip_in: int,
        emb_size_trip_out: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        symmetric_mp: bool = True,
        swap_output: bool = True,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        self.symmetric_mp = symmetric_mp
        self.swap_output = swap_output

        self.dense_ba = Dense(emb_size_in, emb_size_in, activation=activation, bias=False)

        self.mlp_rbf = Dense(emb_size_rbf, emb_size_in, activation=None, bias=False)
        self.scale_rbf = ScaleFactor()

        self.mlp_cbf = EfficientInteractionBilinear(emb_size_trip_in, emb_size_cbf, emb_size_trip_out)
        self.scale_cbf_sum = ScaleFactor()

        self.down_projection = Dense(emb_size_in, emb_size_trip_in, activation=activation, bias=False)
        self.up_projection_ca = Dense(emb_size_trip_out, emb_size_out, activation=activation, bias=False)
        if self.symmetric_mp:
            self.up_projection_ac = Dense(emb_size_trip_out, emb_size_out, activation=activation, bias=False)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        m: torch.Tensor,
        bases: dict[str, torch.Tensor],
        idx: dict[str, torch.Tensor],
        id_swap: torch.Tensor,
        expand_idx: torch.Tensor | None = None,
        idx_agg2: torch.Tensor | None = None,
        idx_agg2_inner: torch.Tensor | None = None,
        agg2_out_size: int | None = None,
    ) -> torch.Tensor:
        """Apply triplet interaction to edge or atom embeddings.

        Args:
            m: Input embeddings (edge or atom), shape
                ``(N, emb_size_in)``.
            bases: Dictionary with keys ``"rad"`` and ``"cir"`` holding
                the pre-computed radial and circular bases.
            idx: Triplet index dictionary with keys ``"in"`` and ``"out"``.
            id_swap: Edge-swap index, shape ``(nEdges,)``.
            expand_idx: Optional expansion index for atom-to-edge paths.
            idx_agg2: Optional second aggregation target index.
            idx_agg2_inner: Optional per-target enumeration for second
                aggregation.
            agg2_out_size: Output size for the second aggregation.

        Returns:
            Updated embeddings of shape ``(N_out, emb_size_out)``.
        """
        x_ba = self.dense_ba(m)
        if expand_idx is not None:
            x_ba = x_ba[expand_idx]

        rad_emb = self.mlp_rbf(bases["rad"])
        x_ba2 = x_ba * rad_emb
        x_ba = self.scale_rbf(x_ba2, ref=x_ba)

        x_ba = self.down_projection(x_ba)
        x_ba = x_ba[idx["in"]]

        x = self.mlp_cbf(
            basis=bases["cir"],
            m=x_ba,
            idx_agg_outer=idx["out"],
            idx_agg_inner=idx["out_agg"],
            idx_agg2_outer=idx_agg2,
            idx_agg2_inner=idx_agg2_inner,
            agg2_out_size=agg2_out_size,
        )
        x = self.scale_cbf_sum(x, ref=x_ba)

        if self.symmetric_mp:
            x_ca = self.up_projection_ca(x)
            x_ac = self.up_projection_ac(x)
            x_ac = x_ac[id_swap]
            x_res = x_ca + x_ac
            return x_res * self.inv_sqrt_2
        else:
            if self.swap_output:
                x = x[id_swap]
            return self.up_projection_ca(x)


class PairInteraction(torch.nn.Module):
    """Atom-pair (A2A) interaction via radial-basis weighted bilinear projection.

    Args:
        emb_size_atom: Atom embedding dimension.
        emb_size_pair_in: Intermediate pair input embedding dimension.
        emb_size_pair_out: Intermediate pair output embedding dimension.
        emb_size_rbf: Radial basis embedding dimension.
        activation: Activation function name.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_pair_in: int,
        emb_size_pair_out: int,
        emb_size_rbf: int,
        activation: str | None = None,
    ) -> None:
        super().__init__()

        self.bilinear = Dense(emb_size_rbf * emb_size_pair_in, emb_size_pair_out, activation=None, bias=False)
        self.scale_rbf_sum = ScaleFactor()

        self.down_projection = Dense(emb_size_atom, emb_size_pair_in, activation=activation, bias=False)
        self.up_projection = Dense(emb_size_pair_out, emb_size_atom, activation=activation, bias=False)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        h: torch.Tensor,
        rad_basis: torch.Tensor,
        edge_index: torch.Tensor,
        target_neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply atom-pair interaction to atom embeddings.

        Args:
            h: Atom embeddings, shape ``(nAtoms, emb_size_atom)``.
            rad_basis: Atom-pair radial basis packed per destination atom,
                shape ``(nAtoms, emb_size_rbf, Kmax)`` where ``Kmax`` is the
                maximum number of neighbours for any atom in the batch.
            edge_index: Atom-pair edge index, shape ``(2, nEdges)``.
            target_neighbor_idx: Per-target atom neighbour enumeration,
                shape ``(nEdges,)``.

        Returns:
            Updated atom embeddings of shape
            ``(nAtoms, emb_size_atom)``.
        """
        num_atoms = h.shape[0]

        x_b = self.down_projection(h)
        x_ba = x_b[edge_index[0]]

        Kmax = 0 if target_neighbor_idx.numel() == 0 else int(target_neighbor_idx.max().item()) + 1
        x2 = x_ba.new_zeros(num_atoms, Kmax, x_ba.shape[-1])
        x2[edge_index[1], target_neighbor_idx] = x_ba

        x_ba2 = rad_basis @ x2
        h_out = self.bilinear(x_ba2.reshape(num_atoms, -1))

        h_out = self.scale_rbf_sum(h_out, ref=x_ba)
        return self.up_projection(h_out)
