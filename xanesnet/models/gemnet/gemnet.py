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

import logging

import torch
from torch_geometric.utils import scatter

from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry
from .layers.atom_update import OutputBlock
from .layers.base import Dense
from .layers.basis import BesselBasisLayer, SphericalBasisLayer, TensorBasisLayer
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding import AtomEmbedding, EdgeEmbedding
from .layers.interaction import InteractionBlock, InteractionBlockTripletsOnly
from .layers.scaling import AutomaticFit


@ModelRegistry.register("gemnet")
class GemNet(Model):
    """The universal directional graph neural network GemNet:
    `"GemNet: Universal Directional Graph Neural Networks for Molecules"`;
    Arxiv: `<https://arxiv.org/abs/2106.08903>`;
    Implementation similar to `<https://github.com/TUM-DAML/gemnet_pytorch/tree/master>`

    Notes:
    - removed forces support
    """

    def __init__(
        self,
        model_type: str,
        # params:
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_quad: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        emb_size_bil_quad: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        triplets_only: bool,
        num_targets: int,
        cutoff: float,
        int_cutoff: float,
        envelope_exponent: int,
        readout: str,  # "add" | "mean"
        output_init: str,
        activation: str,
        scale_file: str | None,
    ) -> None:
        super().__init__(model_type)

        AutomaticFit.reset()

        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.num_blocks = num_blocks
        self.emb_size_atom = emb_size_atom
        self.emb_size_edge = emb_size_edge
        self.emb_size_trip = emb_size_trip
        self.emb_size_quad = emb_size_quad
        self.emb_size_rbf = emb_size_rbf
        self.emb_size_cbf = emb_size_cbf
        self.emb_size_sbf = emb_size_sbf
        self.emb_size_bil_quad = emb_size_bil_quad
        self.emb_size_bil_trip = emb_size_bil_trip
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        self.num_concat = num_concat
        self.num_atom = num_atom
        self.triplets_only = triplets_only
        self.num_targets = num_targets
        self.cutoff = cutoff
        self.int_cutoff = int_cutoff
        self.envelope_exponent = envelope_exponent
        self.readout = readout
        self.output_init = output_init
        self.activation = activation
        self.scale_file = scale_file

        # Basis functions
        self.rbf_basis = BesselBasisLayer(num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)

        if not self.triplets_only:
            self.cbf_basis = SphericalBasisLayer(
                num_spherical,
                num_radial,
                cutoff=int_cutoff,
                envelope_exponent=envelope_exponent,
                efficient=False,
            )
            self.sbf_basis = TensorBasisLayer(
                num_spherical,
                num_radial,
                cutoff=cutoff,
                envelope_exponent=envelope_exponent,
                efficient=True,
            )

        self.cbf_basis3 = SphericalBasisLayer(
            num_spherical,
            num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            efficient=True,
        )

        # Share down projection across all interaction blocks
        if not self.triplets_only:
            self.mlp_rbf4 = Dense(
                num_radial,
                emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf4 = Dense(
                num_radial * num_spherical,
                emb_size_cbf,
                activation=None,
                bias=False,
            )
            self.mlp_sbf4 = EfficientInteractionDownProjection(num_spherical**2, num_radial, emb_size_sbf)
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(num_spherical, num_radial, emb_size_cbf)

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )

        # Embeddings
        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.edge_emb = EdgeEmbedding(emb_size_atom, num_radial, emb_size_edge, activation=activation)

        # Interactions
        int_blocks = []
        interaction_block = (
            InteractionBlockTripletsOnly if self.triplets_only else InteractionBlock
        )  # GemNet-(d)Q or -(d)T
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_quad=emb_size_quad,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_sbf=emb_size_sbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    emb_size_bil_quad=emb_size_bil_quad,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    scale_file=scale_file,
                    name=f"IntBlock_{i+1}",
                )
            )
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        # Output blocks
        out_blocks = []
        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    scale_file=scale_file,
                    name=f"OutBlock_{i}",
                )
            )
        self.out_blocks = torch.nn.ModuleList(out_blocks)

    @staticmethod
    def calculate_interatomic_vectors(
        R: torch.Tensor,  # (nAtoms,3) Atom positions.
        id_s: torch.Tensor,  # (nEdges,) Indices of the source atom of the edges.
        id_t: torch.Tensor,  # (nEdges,) Indices of the target atom of the edges.
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Rt = R[id_t]
        Rs = R[id_s]
        V_st = Rt - Rs  # s -> t
        D_st = torch.sqrt(torch.sum(V_st**2, dim=1))
        V_st = V_st / D_st[..., None]
        return D_st, V_st

    @staticmethod
    def calculate_neighbor_angles(
        R_ac: torch.Tensor,  # (N,3) Vector from atom a to c.
        R_ab: torch.Tensor,  # (N,3) Vector from atom a to b.
    ) -> torch.Tensor:
        # cos(alpha) = (u * v) / (|u|*|v|)
        x = torch.sum(R_ac * R_ab, dim=1)  # shape = (N,)
        # sin(alpha) = |u x v| / (|u|*|v|)
        y = torch.linalg.cross(R_ac, R_ab).norm(dim=-1)  # shape = (N,)
        # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
        y = torch.max(y, torch.tensor(1e-9))
        angle = torch.atan2(y, x)
        return angle

    @staticmethod
    def vector_rejection(
        R_ab: torch.Tensor,  # (N,3) Vector from atom a to b.
        P_n: torch.Tensor,  # (N,3) Normal vector of a plane onto which to project R_ab.
    ) -> torch.Tensor:
        # Project the vector R_ab onto a plane with normal vector P_n.
        a_x_b = torch.sum(R_ab * P_n, dim=-1)
        b_x_b = torch.sum(P_n * P_n, dim=-1)
        return R_ab - (a_x_b / b_x_b)[:, None] * P_n  # (N,3) Projected vector (orthogonal to P_n).

    @staticmethod
    def calculate_angles(
        R: torch.Tensor,  # (nAtoms,3) Atom positions.
        id_c: torch.Tensor,  # (nEdges,) Indices of atom c (source atom of edge).
        id_a: torch.Tensor,  # (nEdges,) Indices of atom a (target atom of edge).
        id4_int_b: torch.Tensor,  # (nInterEdges,) Indices of the atom b of the interaction edge.
        id4_int_a: torch.Tensor,  # (nInterEdges,) Indices of the atom a of the interaction edge.
        id4_expand_abd: torch.Tensor,  # (nQuadruplets,) Indices to map from intermediate d->b to quadruplet d->b.
        id4_reduce_cab: torch.Tensor,  # (nQuadruplets,)  Indices to map from intermediate c->a to quadruplet c->a.
        id4_expand_intm_db: torch.Tensor,  # (intmTriplets,) Indices to map d->b to intermediate d->b.
        id4_reduce_intm_ca: torch.Tensor,  # (intmTriplets,) Indices to map c->a to intermediate c->a.
        id4_expand_intm_ab: torch.Tensor,  # (intmTriplets,) Indices to map b-a to intermediate b-a of the quadruplet's part a-b<-d.
        id4_reduce_intm_ab: torch.Tensor,  # (intmTriplets,) Indices to map b-a to intermediate b-a of the quadruplet's part c->a-b.
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate angles for quadruplet-based message passing.
        # ---------------------------------- a - b <- d ---------------------------------- #
        Ra = R[id4_int_a[id4_expand_intm_ab]]  # a (intmTriplets,3)
        Rb = R[id4_int_b[id4_expand_intm_ab]]  # b (intmTriplets,3)
        Rd = R[id_c[id4_expand_intm_db]]  # d (intmTriplets,3)

        R_ba = Ra - Rb  # (intmTriplets,3)
        R_bd = Rd - Rb  # (intmTriplets,3)
        angle_abd = GemNet.calculate_neighbor_angles(R_ba, R_bd)  # (intmTriplets,)

        # project for calculating gamma
        R_bd_proj = GemNet.vector_rejection(R_bd, R_ba)  # a - b -| d
        R_bd_proj = R_bd_proj[id4_expand_abd]  # (nQuadruplets,)

        # --------------------------------- c -> a <- b ---------------------------------- #
        Rc = R[id_c[id4_reduce_intm_ca]]  # c (intmTriplets,3)
        Ra = R[id_a[id4_reduce_intm_ca]]  # a (intmTriplets,3)
        Rb = R[id4_int_b[id4_reduce_intm_ab]]  # b (intmTriplets,3)

        R_ac = Rc - Ra  # (intmTriplets,3)
        R_ab = Rb - Ra  # (intmTriplets,3)
        angle_cab = GemNet.calculate_neighbor_angles(R_ab, R_ac)  # (intmTriplets,)
        angle_cab = angle_cab[id4_reduce_cab]  # (nQuadruplets,)

        # project for calculating gamma
        R_ac_proj = GemNet.vector_rejection(R_ac, R_ab)  # c |- a - b
        R_ac_proj = R_ac_proj[id4_reduce_cab]  # (nQuadruplets,)

        # -------------------------------- c -> a - b <- d -------------------------------- #
        angle_cabd = GemNet.calculate_neighbor_angles(R_ac_proj, R_bd_proj)  # (nQuadruplets,)

        # RETURN
        # angle_cab: Angle between atoms c <- a -> b.
        # angle_abd: Angle between atoms a <- b -> d.
        # angle_cabd: Angle between atoms c <- a-b -> d.

        return angle_cab, angle_abd, angle_cabd

    @staticmethod
    def calculate_angles3(
        R: torch.Tensor,  # (nAtoms,3) Atom positions.
        id_c: torch.Tensor,  # (nEdges,) Indices of atom c (source atom of edge).
        id_a: torch.Tensor,  # (nEdges,) Indices of atom a (target atom of edge).
        id3_reduce_ca: torch.Tensor,  # (nTriplets,) Edge indices of edge c -> a of the triplets.
        id3_expand_ba: torch.Tensor,  #  (nTriplets,) Edge indices of edge b -> a of the triplets.
    ) -> torch.Tensor:
        # Calculate angles for triplet-based message passing.

        Rc = R[id_c[id3_reduce_ca]]
        Ra = R[id_a[id3_reduce_ca]]
        Rb = R[id_c[id3_expand_ba]]

        # difference vectors
        R_ac = Rc - Ra  # shape = (nTriplets,3)
        R_ab = Rb - Ra  # shape = (nTriplets,3)

        # RETURN
        #  angle_cab: Angle between atoms c <- a -> b.
        # angle in triplets
        return GemNet.calculate_neighbor_angles(R_ac, R_ab)  # (nTriplets,)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        id_a: torch.Tensor,
        id_c: torch.Tensor,
        id_swap: torch.Tensor,
        id3_expand_ba: torch.Tensor,
        id3_reduce_ca: torch.Tensor,
        batch_seg: torch.Tensor,
        Kidx3: torch.Tensor,
        # only if not triplets_only:
        Kidx4: torch.Tensor | None = None,
        id4_int_b: torch.Tensor | None = None,
        id4_int_a: torch.Tensor | None = None,
        id4_reduce_ca: torch.Tensor | None = None,
        id4_reduce_cab: torch.Tensor | None = None,
        id4_expand_abd: torch.Tensor | None = None,
        id4_reduce_intm_ca: torch.Tensor | None = None,
        id4_expand_intm_db: torch.Tensor | None = None,
        id4_reduce_intm_ab: torch.Tensor | None = None,
        id4_expand_intm_ab: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Calculate distances
        D_ca, _ = self.calculate_interatomic_vectors(pos, id_c, id_a)

        cbf4: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
        sbf4: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None

        if not self.triplets_only:
            assert id4_int_b is not None
            assert id4_int_a is not None
            D_ab, _ = self.calculate_interatomic_vectors(pos, id4_int_b, id4_int_a)

            # Calculate angles
            assert id4_expand_abd is not None
            assert id4_reduce_cab is not None
            assert id4_expand_intm_db is not None
            assert id4_reduce_intm_ca is not None
            assert id4_expand_intm_ab is not None
            assert id4_reduce_intm_ab is not None
            Phi_cab, Phi_abd, Theta_cabd = self.calculate_angles(
                pos,
                id_c,
                id_a,
                id4_int_b,
                id4_int_a,
                id4_expand_abd,
                id4_reduce_cab,
                id4_expand_intm_db,
                id4_reduce_intm_ca,
                id4_expand_intm_ab,
                id4_reduce_intm_ab,
            )

            assert Kidx4 is not None
            assert id4_reduce_ca is not None
            cbf4 = self.cbf_basis(D_ab, Phi_abd, id4_expand_intm_ab, None)
            sbf4 = self.sbf_basis(D_ca, Phi_cab, Theta_cabd, id4_reduce_ca, Kidx4)

        rbf = self.rbf_basis(D_ca)
        # Triplet Interaction
        Angles3_cab = self.calculate_angles3(pos, id_c, id_a, id3_reduce_ca, id3_expand_ba)
        cbf3 = self.cbf_basis3(D_ca, Angles3_cab, id3_reduce_ca, Kidx3)

        # Embedding block
        h = self.atom_emb(z)  # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, id_c, id_a)  # (nEdges, emb_size_edge)

        # Shared Down Projections
        if not self.triplets_only:
            rbf4 = self.mlp_rbf4(rbf)
            cbf4 = self.mlp_cbf4(cbf4)
            sbf4 = self.mlp_sbf4(sbf4)
        else:
            rbf4 = None
            cbf4 = None
            sbf4 = None

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(cbf3)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_a = self.out_blocks[0](h, m, rbf_out, id_a)  # (nAtoms, num_targets)

        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf4=rbf4,
                cbf4=cbf4,
                sbf4=sbf4,
                Kidx4=Kidx4,
                rbf3=rbf3,
                cbf3=cbf3,
                Kidx3=Kidx3,
                id_swap=id_swap,
                id3_expand_ba=id3_expand_ba,
                id3_reduce_ca=id3_reduce_ca,
                id4_reduce_ca=id4_reduce_ca,
                id4_expand_intm_db=id4_expand_intm_db,
                id4_expand_abd=id4_expand_abd,
                rbf_h=rbf_h,
                id_c=id_c,
                id_a=id_a,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E = self.out_blocks[i + 1](h, m, rbf_out, id_a)  # (nAtoms, num_targets)
            E_a += E

        nMolecules = int(torch.max(batch_seg).item()) + 1
        if self.readout == "add":
            E_a = scatter(E_a, batch_seg, dim=0, dim_size=nMolecules, reduce="add")
            # (nMolecules, num_targets)
        elif self.readout == "mean":
            E_a = scatter(E_a, batch_seg, dim=0, dim_size=nMolecules, reduce="mean")
            # (nMolecules, num_targets)
        else:
            raise ValueError(f"Invalid readout: {self.readout}")

        return E_a  # (nMolecules, num_targets)

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """
        Initialise all GemNet sub-module weights by calling ``reset_parameters``
        on every sub-module explicitly.

        GemNet uses its own weight initialization scheme (He orthogonal +
        scaling factors) so the ``weights_init`` and ``bias_init`` arguments
        required by the base class are intentionally ignored.
        """
        logging.warning(
            "GemNet uses custom weight initialization, so weights_init and bias_init arguments are ignored."
        )

        # Basis layers
        self.rbf_basis.reset_parameters()

        # Embeddings
        self.atom_emb.reset_parameters()
        self.edge_emb.reset_parameters()

        # Shared down projections
        self.mlp_rbf3.reset_parameters()
        self.mlp_cbf3.reset_parameters()
        self.mlp_rbf_h.reset_parameters()
        self.mlp_rbf_out.reset_parameters()

        if not self.triplets_only:
            self.mlp_rbf4.reset_parameters()
            self.mlp_cbf4.reset_parameters()
            self.mlp_sbf4.reset_parameters()

        # Interaction blocks
        for block in self.int_blocks:
            block.reset_parameters()

        # Output blocks
        for block in self.out_blocks:
            block.reset_parameters()

    @property
    def signature(self) -> Config:
        """
        Return model signature as a dictionary.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "num_spherical": self.num_spherical,
                "num_radial": self.num_radial,
                "num_blocks": self.num_blocks,
                "emb_size_atom": self.emb_size_atom,
                "emb_size_edge": self.emb_size_edge,
                "emb_size_trip": self.emb_size_trip,
                "emb_size_quad": self.emb_size_quad,
                "emb_size_rbf": self.emb_size_rbf,
                "emb_size_cbf": self.emb_size_cbf,
                "emb_size_sbf": self.emb_size_sbf,
                "emb_size_bil_quad": self.emb_size_bil_quad,
                "emb_size_bil_trip": self.emb_size_bil_trip,
                "num_before_skip": self.num_before_skip,
                "num_after_skip": self.num_after_skip,
                "num_concat": self.num_concat,
                "num_atom": self.num_atom,
                "triplets_only": self.triplets_only,
                "num_targets": self.num_targets,
                "cutoff": self.cutoff,
                "int_cutoff": self.int_cutoff,
                "envelope_exponent": self.envelope_exponent,
                "readout": self.readout,
                "output_init": self.output_init,
                "activation": self.activation,
                "scale_file": self.scale_file,
            }
        )
        return signature
