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

import math

import torch

from .atom_update_block import AtomUpdateBlock
from .base_layers import Dense, ResidualLayer
from .efficient import EfficientInteractionBilinear
from .embedding_block import EdgeEmbedding
from .scaling import ScaleFactor


class InteractionBlock(torch.nn.Module):
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
        activation=None,
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
        h,
        m,
        bases_qint,
        bases_e2e,
        bases_a2e,
        bases_e2a,
        basis_a2a_rad,
        basis_atom_update,
        edge_index_main,
        a2ee2a_graph,
        a2a_graph,
        id_swap,
        trip_idx_e2e,
        trip_idx_a2e,
        trip_idx_e2a,
        quad_idx,
    ):
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
    def __init__(
        self,
        emb_size_edge,
        emb_size_quad_in,
        emb_size_quad_out,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_sbf,
        symmetric_mp=True,
        activation=None,
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

    def forward(self, m, bases, idx, id_swap):
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
        activation=None,
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
        m,
        bases,
        idx,
        id_swap,
        expand_idx=None,
        idx_agg2=None,
        idx_agg2_inner=None,
        agg2_out_size=None,
    ):
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
    def __init__(self, emb_size_atom, emb_size_pair_in, emb_size_pair_out, emb_size_rbf, activation=None) -> None:
        super().__init__()

        self.bilinear = Dense(emb_size_rbf * emb_size_pair_in, emb_size_pair_out, activation=None, bias=False)
        self.scale_rbf_sum = ScaleFactor()

        self.down_projection = Dense(emb_size_atom, emb_size_pair_in, activation=activation, bias=False)
        self.up_projection = Dense(emb_size_pair_out, emb_size_atom, activation=activation, bias=False)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, h, rad_basis, edge_index, target_neighbor_idx):
        num_atoms = h.shape[0]

        x_b = self.down_projection(h)
        x_ba = x_b[edge_index[0]]

        Kmax = torch.max(target_neighbor_idx) + 1
        x2 = x_ba.new_zeros(num_atoms, Kmax, x_ba.shape[-1])
        x2[edge_index[1], target_neighbor_idx] = x_ba

        x_ba2 = rad_basis @ x2
        h_out = self.bilinear(x_ba2.reshape(num_atoms, -1))

        h_out = self.scale_rbf_sum(h_out, ref=x_ba)
        return self.up_projection(h_out)
