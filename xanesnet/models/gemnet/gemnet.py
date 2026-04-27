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

"""GemNet universal directional graph neural network for molecular property prediction."""

import logging

import torch

from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry
from .layers.atom_update import OutputBlock
from .layers.base import Dense
from .layers.basis import BesselBasisLayer, SphericalBasisLayer, TensorBasisLayer
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding import AtomEmbedding, EdgeEmbedding
from .layers.interaction import InteractionBlock, InteractionBlockTripletsOnly
from .layers.scaling import load_scales_json


@ModelRegistry.register("gemnet")
class GemNet(Model):
    """Universal directional graph neural network (GemNet).

    Reference: `GemNet: Universal Directional Graph Neural Networks for Molecules
    <https://arxiv.org/abs/2106.08903>`_.

    Implementation based on
    `TUM-DAML/gemnet_pytorch <https://github.com/TUM-DAML/gemnet_pytorch/tree/master>`_.

    Args:
        model_type: Model type string (passed to base class).
        num_spherical: Number of spherical harmonics / basis functions.
        num_radial: Number of radial basis functions.
        num_blocks: Number of interaction blocks.
        emb_size_atom: Embedding size for atoms.
        emb_size_edge: Embedding size for edges.
        emb_size_trip: Down-projected embedding size in the triplet interaction block.
        emb_size_quad: Down-projected embedding size in the quadruplet interaction block.
        emb_size_rbf: Embedding size of the radial basis projection.
        emb_size_cbf: Embedding size of the circular basis projection.
        emb_size_sbf: Embedding size of the spherical basis projection.
        emb_size_bil_quad: Edge embedding size after the bilinear layer in the quadruplet block.
        emb_size_bil_trip: Edge embedding size after the bilinear layer in the triplet block.
        num_before_skip: Number of residual blocks before each skip connection.
        num_after_skip: Number of residual blocks after each skip connection.
        num_concat: Number of residual blocks after the atom-edge concatenation.
        num_atom: Number of residual blocks in the atom update block.
        triplets_only: If ``True``, use GemNet-T/dT (triplets only, no quadruplets).
        num_targets: Output dimension per atom.
        cutoff: Main graph edge cutoff in **A**.
        int_cutoff: Interaction graph edge cutoff in **A** (quadruplet variant only).
        envelope_exponent: Exponent ``p`` for the polynomial cutoff envelope.
        output_init: Weight initialisation for the final output layer (``"HeOrthogonal"`` or ``"zeros"``).
        activation: Activation function name (``"swish"`` / ``"silu"``).
        scale_file: Path to a JSON file with pre-fitted variance scale factors, or ``None`` to use the default value of 1.0 for all factors.
        num_elements: Number of distinct element types in the embedding table.
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
        output_init: str,
        activation: str,
        scale_file: str | None,
        num_elements: int,
    ) -> None:
        super().__init__(model_type)

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
        self.output_init = output_init
        self.activation = activation
        self.num_elements = num_elements

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
        self.atom_emb = AtomEmbedding(emb_size_atom, num_elements=num_elements)
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

        # Variance-preserving scale factors
        self.scale_file = scale_file
        if scale_file is not None:
            load_scales_json(self, scale_file)

    @staticmethod
    def calculate_neighbor_angles(
        R_ac: torch.Tensor,
        R_ab: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the angle between two bond vectors at a shared atom.

        Args:
            R_ac: Bond vectors from atom *a* to atom *c*, shape ``(N, 3)``.
            R_ab: Bond vectors from atom *a* to atom *b*, shape ``(N, 3)``.

        Returns:
            Angles in radians, shape ``(N,)``.
        """
        x = torch.sum(R_ac * R_ab, dim=1)  # (N,)
        y = torch.linalg.cross(R_ac, R_ab).norm(dim=-1)  # (N,)
        # Avoid NaN gradient when y == 0.
        y = y.clamp_min(1e-9)
        angle = torch.atan2(y, x)
        return angle

    @staticmethod
    def vector_rejection(
        R_ab: torch.Tensor,
        P_n: torch.Tensor,
    ) -> torch.Tensor:
        """Project ``R_ab`` onto the plane whose normal is ``P_n``.

        Args:
            R_ab: Vectors to project, shape ``(N, 3)``.
            P_n: Normal vectors of the projection plane, shape ``(N, 3)``.

        Returns:
            Projected vectors orthogonal to ``P_n``, shape ``(N, 3)``.
        """
        a_x_b = torch.sum(R_ab * P_n, dim=-1)
        b_x_b = torch.sum(P_n * P_n, dim=-1)
        return R_ab - (a_x_b / b_x_b)[:, None] * P_n  # (N, 3) projected, orthogonal to P_n

    @staticmethod
    def calculate_angles3(
        edge_vec: torch.Tensor,
        id3_reduce_ca: torch.Tensor,
        id3_expand_ba: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet angles for message passing.

        Uses pre-computed PBC-correct edge vectors to avoid NaN for periodic
        self-image edges where ``pos[id_a] == pos[id_c]``.

        Args:
            edge_vec: PBC-aware edge vectors (id_c -> id_a), shape ``(nEdges, 3)``.
            id3_reduce_ca: Edge index of c -> a for each triplet, shape ``(nTriplets,)``.
            id3_expand_ba: Edge index of b -> a for each triplet, shape ``(nTriplets,)``.

        Returns:
            Triplet angles at atom *a*, shape ``(nTriplets,)``.
        """
        R_ac = -edge_vec[id3_reduce_ca]  # a -> c
        R_ab = -edge_vec[id3_expand_ba]  # a -> b
        return GemNet.calculate_neighbor_angles(R_ac, R_ab)

    @staticmethod
    def calculate_angles(
        edge_vec: torch.Tensor,
        int_edge_vec: torch.Tensor,
        id4_expand_abd: torch.Tensor,
        id4_reduce_cab: torch.Tensor,
        id4_expand_intm_db: torch.Tensor,
        id4_reduce_intm_ca: torch.Tensor,
        id4_expand_intm_ab: torch.Tensor,
        id4_reduce_intm_ab: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the three angles required for quadruplet-based message passing.

        Uses PBC-aware pre-computed edge vectors to avoid NaN on periodic
        self-image edges. The three returned angles correspond to:

        * ``angle_cab``: angle at atom *a* between bonds c <- a -> b.
        * ``angle_abd``: angle at atom *b* between bonds a <- b -> d.
        * ``angle_cabd``: dihedral-like angle between the projected planes
          c <- a-b -> d.

        Args:
            edge_vec: Main-graph edge vectors (id_c -> id_a), shape
                ``(nEdges, 3)``.
            int_edge_vec: Interaction-graph edge vectors (id4_int_b -> id4_int_a),
                shape ``(nIntEdges, 3)``.
            id4_expand_abd: Quadruplet index mapping to the a-b-d triplet.
            id4_reduce_cab: Quadruplet index mapping to the c-a-b triplet.
            id4_expand_intm_db: Intermediate-triplet index for the d -> b edge.
            id4_reduce_intm_ca: Intermediate-triplet index for the c -> a edge.
            id4_expand_intm_ab: Intermediate-triplet index for the a -> b edge.
            id4_reduce_intm_ab: Intermediate-triplet index for the a -> b edge
                (reduce side).

        Returns:
            Tuple ``(angle_cab, angle_abd, angle_cabd)``, each of shape
            ``(nQuadruplets,)``.
        """
        # ---------------------------------- a - b <- d ---------------------------------- #
        # int_edge_vec (source b -> target a): R_ba = Ra - Rb.
        R_ba = int_edge_vec[id4_expand_intm_ab]  # (intmTriplets, 3)
        # Main edge d->b: edge_vec = Rb - Rd, so R_bd = -edge_vec.
        R_bd = -edge_vec[id4_expand_intm_db]  # (intmTriplets, 3)
        angle_abd = GemNet.calculate_neighbor_angles(R_ba, R_bd)

        R_bd_proj = GemNet.vector_rejection(R_bd, R_ba)
        R_bd_proj = R_bd_proj[id4_expand_abd]

        # --------------------------------- c -> a <- b ---------------------------------- #
        # Main edge c->a: edge_vec = Ra - Rc, so R_ac = -edge_vec.
        R_ac = -edge_vec[id4_reduce_intm_ca]
        # Interaction edge b->a: int_edge_vec = Ra - Rb, so R_ab = -int_edge_vec.
        R_ab = -int_edge_vec[id4_reduce_intm_ab]
        angle_cab = GemNet.calculate_neighbor_angles(R_ab, R_ac)
        angle_cab = angle_cab[id4_reduce_cab]

        R_ac_proj = GemNet.vector_rejection(R_ac, R_ab)
        R_ac_proj = R_ac_proj[id4_reduce_cab]

        # -------------------------------- c -> a - b <- d -------------------------------- #
        # angle_cab: Angle between atoms c <- a -> b.
        # angle_abd: Angle between atoms a <- b -> d.
        # angle_cabd: Angle between atoms c <- a-b -> d.
        angle_cabd = GemNet.calculate_neighbor_angles(R_ac_proj, R_bd_proj)

        return angle_cab, angle_abd, angle_cabd

    def forward(
        self,
        z: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_weight: torch.Tensor,
        id_a: torch.Tensor,
        id_c: torch.Tensor,
        id_swap: torch.Tensor,
        id3_expand_ba: torch.Tensor,
        id3_reduce_ca: torch.Tensor,
        Kidx3: torch.Tensor,
        # only if not triplets_only:
        int_edge_vec: torch.Tensor | None = None,
        int_edge_weight: torch.Tensor | None = None,
        Kidx4: torch.Tensor | None = None,
        id4_reduce_ca: torch.Tensor | None = None,
        id4_reduce_cab: torch.Tensor | None = None,
        id4_expand_abd: torch.Tensor | None = None,
        id4_reduce_intm_ca: torch.Tensor | None = None,
        id4_expand_intm_db: torch.Tensor | None = None,
        id4_reduce_intm_ab: torch.Tensor | None = None,
        id4_expand_intm_ab: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the GemNet forward pass and return per-atom target predictions.

        Args:
            z: Atomic numbers, shape ``(nAtoms,)``.
            edge_vec: PBC-aware main-graph edge vectors (id_c -> id_a),
                shape ``(nEdges, 3)``.
            edge_weight: Main-graph edge lengths (distances) in **A**,
                shape ``(nEdges,)``.
            id_a: Target atom index for each edge (c -> a), shape ``(nEdges,)``.
            id_c: Source atom index for each edge (c -> a), shape ``(nEdges,)``.
            id_swap: Index mapping each edge to its reverse, shape ``(nEdges,)``.
            id3_expand_ba: Edge index of b -> a for each triplet,
                shape ``(nTriplets,)``.
            id3_reduce_ca: Edge index of c -> a for each triplet,
                shape ``(nTriplets,)``.
            Kidx3: Neighbour index within the sparse dense matrix for triplets,
                shape ``(nTriplets,)``.
            int_edge_vec: Interaction-graph edge vectors, shape ``(nIntEdges, 3)``.
                Required when ``triplets_only=False``.
            int_edge_weight: Interaction-graph edge lengths in **A**,
                shape ``(nIntEdges,)``.
                Required when ``triplets_only=False``.
            Kidx4: Neighbour index for the quadruplet sparse dense matrix.
                Required when ``triplets_only=False``.
            id4_reduce_ca: Quadruplet edge index c -> a.
                Required when ``triplets_only=False``.
            id4_reduce_cab: Quadruplet reduce index for c-a-b.
                Required when ``triplets_only=False``.
            id4_expand_abd: Quadruplet expand index for a-b-d.
                Required when ``triplets_only=False``.
            id4_reduce_intm_ca: Intermediate-triplet index c -> a.
                Required when ``triplets_only=False``.
            id4_expand_intm_db: Intermediate-triplet index d -> b.
                Required when ``triplets_only=False``.
            id4_reduce_intm_ab: Intermediate-triplet index a -> b (reduce side).
                Required when ``triplets_only=False``.
            id4_expand_intm_ab: Intermediate-triplet index a -> b (expand side).
                Required when ``triplets_only=False``.

        Returns:
            Per-atom target predictions of shape ``(nAtoms, num_targets)``.
        """
        D_ca = edge_weight

        cbf4: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
        sbf4: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None

        if not self.triplets_only:
            assert int_edge_vec is not None
            assert int_edge_weight is not None
            D_ab = int_edge_weight

            # Calculate angles
            assert id4_expand_abd is not None
            assert id4_reduce_cab is not None
            assert id4_expand_intm_db is not None
            assert id4_reduce_intm_ca is not None
            assert id4_expand_intm_ab is not None
            assert id4_reduce_intm_ab is not None
            Phi_cab, Phi_abd, Theta_cabd = self.calculate_angles(
                edge_vec,
                int_edge_vec,
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
        Angles3_cab = self.calculate_angles3(edge_vec, id3_reduce_ca, id3_expand_ba)
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

        return E_a  # (nAtoms, num_targets)

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """Initialise GemNet weights using its built-in He-orthogonal scheme.

        GemNet uses its own weight initialisation (He-orthogonal + variance-
        preserving scale factors), so the ``weights_init`` and ``bias_init``
        arguments required by the base-class interface are intentionally ignored.
        A warning is emitted when called.

        Args:
            weights_init: Ignored. Present for base-class interface compatibility.
            bias_init: Ignored. Present for base-class interface compatibility.
            **kwargs: Additional keyword arguments (ignored).
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
        """Return the model signature as a :class:`~xanesnet.serialization.config.Config`.

        Note:
            ``scale_file`` is always set to ``None`` in the signature. The fitted
            scale-factor values are carried by the model ``state_dict`` as
            non-trainable parameters, making saved checkpoints fully portable.
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
                "output_init": self.output_init,
                "activation": self.activation,
                # scale_file is intentionally None in the signature: the actual fitted
                # values are carried by state_dict (ScaleFactor parameters), making
                # the saved model fully self-contained and portable across machines.
                "scale_file": None,
                "num_elements": self.num_elements,
            }
        )
        return signature
