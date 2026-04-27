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

"""GemNet-OC universal directional graph neural network for molecular property prediction."""

import logging

import torch
from torch import nn

from xanesnet.models.base import Model
from xanesnet.models.registry import ModelRegistry
from xanesnet.serialization.config import Config

from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense, ResidualLayer
from .layers.efficient import BasisEmbedding
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.interaction_block import InteractionBlock
from .layers.radial_basis import RadialBasis
from .layers.scaling import load_scales_json
from .layers.spherical_basis import CircularBasisLayer, SphericalBasisLayer
from .utils import get_angle, get_initializer, get_inner_idx, inner_product_clamped


@ModelRegistry.register("gemnet_oc")
class GemNetOC(Model):
    """GemNet-OC adapted for per-atom XANES spectrum prediction.

    Ported from the fairchem-core reference (MIT License).

    Note:
        **Periodicity handling**: This model is agnostic to periodicity. It
        consumes precomputed, PBC-aware tensors produced by
        :class:`~xanesnet.datasets.torchgeometric.gemnet.GemNetDataset`. The
        dataset builds each graph with ``build_edges`` on a ``pymatgen``
        object (which handles periodic and non-periodic inputs uniformly) and
        emits lattice-corrected vectors for periodic self-image edges. The
        model never reads raw ``batch.pos`` and never recomputes distances or
        vectors.

    Note:
        **Basis-function configuration**: ``rbf``, ``rbf_spherical``,
        ``envelope``, ``cbf``, and ``sbf`` are
        :class:`~xanesnet.serialization.config.Config` objects. Each carries
        a required ``name`` key plus optional hyperparameters. Programmatic
        callers must wrap raw dicts in ``Config(...)`` before passing them in.

    Args:
            model_type: Registry/model identifier passed to the base
                :class:`~xanesnet.models.base.Model`.
            num_targets: Number of output spectral channels predicted per atom.
            num_spherical: Number of angular basis functions.
            num_radial: Number of radial basis functions.
            num_blocks: Number of interaction blocks.
            emb_size_atom: Atom embedding dimension.
            emb_size_edge: Edge embedding dimension.
            emb_size_trip_in: Intermediate triplet input dimension.
            emb_size_trip_out: Intermediate triplet output dimension.
            emb_size_quad_in: Intermediate quadruplet input dimension.
            emb_size_quad_out: Intermediate quadruplet output dimension.
            emb_size_aint_in: Intermediate atom-pair input dimension.
            emb_size_aint_out: Intermediate atom-pair output dimension.
            emb_size_rbf: Embedded radial-basis dimension.
            emb_size_cbf: Embedded circular-basis dimension.
            emb_size_sbf: Embedded spherical-basis dimension.
            num_before_skip: Number of residual layers before the edge skip
                connection inside each interaction block.
            num_after_skip: Number of residual layers after the edge skip
                connection inside each interaction block.
            num_concat: Number of residual layers after the atom/edge concat
                update in each interaction block.
            num_atom: Number of residual layers in atom update and output blocks.
            num_output_afteratom: Number of residual layers applied after the
                atom skip connection inside each output block.
            num_atom_emb_layers: Number of extra atom residual layers inside
                each interaction block.
            num_global_out_layers: Number of residual layers in the final per-atom output MLP.
            cutoff: Main graph cutoff radius in **angstrom**.
            cutoff_qint: Optional quadruplet cutoff radius. Uses ``cutoff``
                when ``None`` or when quadruplet interactions are disabled.
            cutoff_aeaint: Optional atom-edge / edge-atom cutoff radius. Uses
                ``cutoff`` when ``None`` or when those interaction paths are
                disabled.
            cutoff_aint: Optional atom-atom cutoff radius. Uses the maximum of
                the active cutoffs when ``None`` or when atom-pair interactions are disabled.
            rbf: Config for the main radial basis.
            rbf_spherical: Config for the radial basis used together with
                circular/spherical angular bases.
            envelope: Config for the cutoff envelope applied to radial bases.
            cbf: Config for the circular basis.
            sbf: Config for the spherical basis.
            output_init: Initialiser name for the final linear output layer.
            activation: Activation name used in dense and residual layers.
            quad_interaction: Whether to enable quadruplet interactions.
            atom_edge_interaction: Whether to enable atom-to-edge interactions.
            edge_atom_interaction: Whether to enable edge-to-atom interactions.
            atom_interaction: Whether to enable atom-to-atom interactions.
            scale_basis: Whether to apply learnable scale factors to the basis functions.
            num_elements: Size of the atomic-number embedding table.
            scale_file: Optional JSON file with pre-fitted scale factors.
    """

    def __init__(
        self,
        model_type: str,
        # params:
        num_targets: int,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip_in: int,
        emb_size_trip_out: int,
        emb_size_quad_in: int,
        emb_size_quad_out: int,
        emb_size_aint_in: int,
        emb_size_aint_out: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        num_output_afteratom: int,
        num_atom_emb_layers: int,
        num_global_out_layers: int,
        cutoff: float,
        cutoff_qint: float | None,
        cutoff_aeaint: float | None,
        cutoff_aint: float | None,
        rbf: Config,
        rbf_spherical: Config,
        envelope: Config,
        cbf: Config,
        sbf: Config,
        output_init: str,
        activation: str,
        quad_interaction: bool,
        atom_edge_interaction: bool,
        edge_atom_interaction: bool,
        atom_interaction: bool,
        scale_basis: bool,
        num_elements: int,
        scale_file: str | None = None,
    ) -> None:
        """Initialise the GemNet-OC XANES model."""
        super().__init__(model_type)
        self.num_blocks = num_blocks
        self.num_targets = num_targets
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_atom = emb_size_atom
        self.emb_size_edge = emb_size_edge
        self.emb_size_trip_in = emb_size_trip_in
        self.emb_size_trip_out = emb_size_trip_out
        self.emb_size_quad_in = emb_size_quad_in
        self.emb_size_quad_out = emb_size_quad_out
        self.emb_size_aint_in = emb_size_aint_in
        self.emb_size_aint_out = emb_size_aint_out
        self.emb_size_rbf = emb_size_rbf
        self.emb_size_cbf = emb_size_cbf
        self.emb_size_sbf = emb_size_sbf
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        self.num_concat = num_concat
        self.num_atom = num_atom
        self.num_output_afteratom = num_output_afteratom
        self.num_atom_emb_layers = num_atom_emb_layers
        self.num_global_out_layers = num_global_out_layers

        self.rbf_cfg = rbf
        self.rbf_spherical_cfg = rbf_spherical
        self.envelope_cfg = envelope
        self.cbf_cfg = cbf
        self.sbf_cfg = sbf
        self.output_init = output_init
        self.scale_basis = scale_basis
        self.num_elements = num_elements
        self.activation = activation
        self.quad_interaction = quad_interaction
        self.atom_edge_interaction = atom_edge_interaction
        self.edge_atom_interaction = edge_atom_interaction
        self.atom_interaction = atom_interaction

        self._set_cutoffs(cutoff, cutoff_qint, cutoff_aeaint, cutoff_aint)

        self._init_basis_functions(
            num_radial,
            num_spherical,
            self.rbf_cfg,
            self.rbf_spherical_cfg,
            self.envelope_cfg,
            self.cbf_cfg,
            self.sbf_cfg,
            scale_basis,
        )
        self._init_shared_basis_layers(num_radial, num_spherical, emb_size_rbf, emb_size_cbf, emb_size_sbf)

        # Embedding blocks
        self.atom_emb = AtomEmbedding(emb_size_atom, num_elements)
        self.edge_emb = EdgeEmbedding(emb_size_atom, num_radial, emb_size_edge, activation=activation)

        # Interaction blocks
        self.int_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip_in=emb_size_trip_in,
                    emb_size_trip_out=emb_size_trip_out,
                    emb_size_quad_in=emb_size_quad_in,
                    emb_size_quad_out=emb_size_quad_out,
                    emb_size_a2a_in=emb_size_aint_in,
                    emb_size_a2a_out=emb_size_aint_out,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_sbf=emb_size_sbf,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    num_atom_emb_layers=num_atom_emb_layers,
                    quad_interaction=quad_interaction,
                    atom_edge_interaction=atom_edge_interaction,
                    edge_atom_interaction=edge_atom_interaction,
                    atom_interaction=atom_interaction,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ]
        )

        # Output blocks (one more than interaction blocks: initial + per-block)
        self.out_blocks = nn.ModuleList(
            [
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    nHidden_afteratom=num_output_afteratom,
                    activation=activation,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        # Global output MLP (concatenated across blocks) -> per-atom spectrum
        out_mlp_E = [Dense(emb_size_atom * (num_blocks + 1), emb_size_atom, activation=activation)] + [
            ResidualLayer(emb_size_atom, activation=activation) for _ in range(num_global_out_layers)
        ]
        self.out_mlp_E = nn.Sequential(*out_mlp_E)
        self.out_energy = Dense(emb_size_atom, num_targets, bias=False, activation=None)

        out_initializer = get_initializer(output_init)
        self.out_energy.reset_parameters(out_initializer)

        # Variance-preserving scale factors
        self.scale_file = scale_file
        if scale_file is not None:
            load_scales_json(self, scale_file)

    def _set_cutoffs(
        self,
        cutoff: float,
        cutoff_qint: float | None,
        cutoff_aeaint: float | None,
        cutoff_aint: float | None,
    ) -> None:
        """Resolve effective cutoff radii for each interaction type.

        Missing cutoffs default to ``cutoff``; ``cutoff_aint`` defaults to the
        maximum of all other cutoffs when not explicitly provided.

        Args:
            cutoff: Primary edge cutoff radius in **angstrom**.
            cutoff_qint: Quadruplet interaction cutoff, or ``None`` to use ``cutoff``.
            cutoff_aeaint: Atom-to-edge / edge-to-atom interaction cutoff, or ``None`` to use ``cutoff``.
            cutoff_aint: Atom-to-atom interaction cutoff, or ``None`` to use
                the maximum of all other cutoffs.
        """
        self.cutoff = cutoff
        if not (self.atom_edge_interaction or self.edge_atom_interaction) or cutoff_aeaint is None:
            self.cutoff_aeaint = cutoff
        else:
            self.cutoff_aeaint = cutoff_aeaint
        if not self.quad_interaction or cutoff_qint is None:
            self.cutoff_qint = cutoff
        else:
            self.cutoff_qint = cutoff_qint
        if not self.atom_interaction or cutoff_aint is None:
            self.cutoff_aint = max(self.cutoff, self.cutoff_aeaint, self.cutoff_qint)
        else:
            self.cutoff_aint = cutoff_aint

        assert self.cutoff <= self.cutoff_aint
        assert self.cutoff_aeaint <= self.cutoff_aint
        assert self.cutoff_qint <= self.cutoff_aint

    def _init_basis_functions(
        self,
        num_radial: int,
        num_spherical: int,
        rbf: Config,
        rbf_spherical: Config,
        envelope: Config,
        cbf: Config,
        sbf: Config,
        scale_basis: bool,
    ) -> None:
        """Instantiate all radial, circular, and spherical basis modules.

        Args:
            num_radial: Number of radial basis functions.
            num_spherical: Number of angular basis functions.
            rbf: Config for the main radial basis.
            rbf_spherical: Config for the radial basis paired with angular bases.
            envelope: Config for the radial envelope.
            cbf: Config for the circular basis.
            sbf: Config for the spherical basis.
            scale_basis: Whether to wrap basis outputs in learnable scale factors.
        """
        self.radial_basis = RadialBasis(
            num_radial=num_radial, cutoff=self.cutoff, rbf=rbf, envelope=envelope, scale_basis=scale_basis
        )
        radial_basis_spherical = RadialBasis(
            num_radial=num_radial, cutoff=self.cutoff, rbf=rbf_spherical, envelope=envelope, scale_basis=scale_basis
        )
        if self.quad_interaction:
            radial_basis_spherical_qint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_qint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_qint = CircularBasisLayer(
                num_spherical, radial_basis=radial_basis_spherical_qint, cbf=cbf, scale_basis=scale_basis
            )
            self.sbf_basis_qint = SphericalBasisLayer(
                num_spherical, radial_basis=radial_basis_spherical, sbf=sbf, scale_basis=scale_basis
            )
        if self.atom_edge_interaction:
            self.radial_basis_aeaint = RadialBasis(
                num_radial=num_radial, cutoff=self.cutoff_aeaint, rbf=rbf, envelope=envelope, scale_basis=scale_basis
            )
            self.cbf_basis_aeint = CircularBasisLayer(
                num_spherical, radial_basis=radial_basis_spherical, cbf=cbf, scale_basis=scale_basis
            )
        if self.edge_atom_interaction:
            if not self.atom_edge_interaction:
                self.radial_basis_aeaint = RadialBasis(
                    num_radial=num_radial,
                    cutoff=self.cutoff_aeaint,
                    rbf=rbf,
                    envelope=envelope,
                    scale_basis=scale_basis,
                )
            radial_basis_spherical_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_eaint = CircularBasisLayer(
                num_spherical, radial_basis=radial_basis_spherical_aeaint, cbf=cbf, scale_basis=scale_basis
            )
        if self.atom_interaction:
            self.radial_basis_aint = RadialBasis(
                num_radial=num_radial, cutoff=self.cutoff_aint, rbf=rbf, envelope=envelope, scale_basis=scale_basis
            )

        self.cbf_basis_tint = CircularBasisLayer(
            num_spherical, radial_basis=radial_basis_spherical, cbf=cbf, scale_basis=scale_basis
        )

    def _init_shared_basis_layers(
        self,
        num_radial: int,
        num_spherical: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
    ) -> None:
        """Instantiate shared basis-projection layers.

        Args:
            num_radial: Number of radial basis functions.
            num_spherical: Number of angular basis functions.
            emb_size_rbf: Embedded radial-basis dimension.
            emb_size_cbf: Embedded circular-basis dimension.
            emb_size_sbf: Embedded spherical-basis dimension.
        """
        if self.quad_interaction:
            self.mlp_rbf_qint = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
            self.mlp_cbf_qint = BasisEmbedding(num_radial, emb_size_cbf, num_spherical)
            self.mlp_sbf_qint = BasisEmbedding(num_radial, emb_size_sbf, num_spherical**2)
        if self.atom_edge_interaction:
            self.mlp_rbf_aeint = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
            self.mlp_cbf_aeint = BasisEmbedding(num_radial, emb_size_cbf, num_spherical)
        if self.edge_atom_interaction:
            self.mlp_rbf_eaint = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
            self.mlp_cbf_eaint = BasisEmbedding(num_radial, emb_size_cbf, num_spherical)
        if self.atom_interaction:
            self.mlp_rbf_aint = BasisEmbedding(num_radial, emb_size_rbf)

        self.mlp_rbf_tint = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_cbf_tint = BasisEmbedding(num_radial, emb_size_cbf, num_spherical)

        self.mlp_rbf_h = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_rbf_out = Dense(num_radial, emb_size_rbf, activation=None, bias=False)

    def _calculate_quad_angles(
        self,
        V_st: torch.Tensor,
        V_qint_st: torch.Tensor,
        quad_idx: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute angle cosines and dihedral angles for quadruplet message passing.

        Args:
            V_st: PBC-aware edge displacement vectors for the main graph,
                shape ``(nEdges, 3)``.
            V_qint_st: PBC-aware edge displacement vectors for the
                quadruplet interaction graph, shape ``(nQintEdges, 3)``.
            quad_idx: Quadruplet index dictionary.

        Returns:
            Tuple ``(cosφ_cab, cosφ_abd, angle_cabd)`` where ``cosφ_cab`` and
            ``cosφ_abd`` are cosine angles and ``angle_cabd`` is a dihedral
            angle in **radians**, all of shape ``(nQuadruplets,)``.
        """
        V_ba = V_qint_st[quad_idx["triplet_in"]["out"]]
        V_db = V_st[quad_idx["triplet_in"]["in"]]
        cosφ_abd = inner_product_clamped(V_ba, V_db)

        V_db_cross = torch.cross(V_db, V_ba, dim=-1)
        V_db_cross = V_db_cross[quad_idx["trip_in_to_quad"]]

        V_ca = V_st[quad_idx["triplet_out"]["out"]]
        V_ba = V_qint_st[quad_idx["triplet_out"]["in"]]
        cosφ_cab = inner_product_clamped(V_ca, V_ba)

        V_ca_cross = torch.cross(V_ca, V_ba, dim=-1)
        V_ca_cross = V_ca_cross[quad_idx["trip_out_to_quad"]]

        angle_cabd = get_angle(V_ca_cross, V_db_cross)
        return cosφ_cab, cosφ_abd, angle_cabd

    def _get_bases(
        self,
        main_graph: dict[str, torch.Tensor],
        a2a_graph: dict[str, torch.Tensor],
        a2ee2a_graph: dict[str, torch.Tensor],
        qint_graph: dict[str, torch.Tensor],
        trip_idx_e2e: dict[str, torch.Tensor],
        trip_idx_a2e: dict[str, torch.Tensor],
        trip_idx_e2a: dict[str, torch.Tensor],
        quad_idx: dict,
        num_atoms: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor | None,
    ]:
        """Compute all radial, circular, and spherical basis tensors.

        Each interaction-type branch computes the raw basis tensors and
        immediately applies the corresponding ``mlp_*`` / :class:`BasisEmbedding`
        layers. Disabled branches contribute empty dicts or ``None``.

        Args:
            main_graph: Main edge graph dict with keys ``"edge_index"``,
                ``"distance"``, ``"vector"``.
            a2a_graph: Atom-to-atom graph dict (may be empty).
            a2ee2a_graph: Atom-to-edge-to-atom graph dict (may be empty).
            qint_graph: Quadruplet interaction graph dict (may be empty).
            trip_idx_e2e: Triplet indices for E2E interactions.
            trip_idx_a2e: Triplet indices for A2E interactions.
            trip_idx_e2a: Triplet indices for E2A interactions.
            quad_idx: Quadruplet index dict (may be empty).
            num_atoms: Total number of atoms in the batch.

        Returns:
            Tuple of eight items:
            ``(basis_rad_main_raw, basis_atom_update, basis_output,
            bases_qint, bases_e2e, bases_a2e, bases_e2a, basis_a2a_rad)``.
        """
        # Main graph: always required.
        basis_rad_main_raw = self.radial_basis(main_graph["distance"])
        basis_atom_update = self.mlp_rbf_h(basis_rad_main_raw)
        basis_output = self.mlp_rbf_out(basis_rad_main_raw)

        # Edge -> edge (triplet) bases: always required.
        cosφ_cab = inner_product_clamped(
            main_graph["vector"][trip_idx_e2e["out"]],
            main_graph["vector"][trip_idx_e2e["in"]],
        )
        rad_cir_e2e, cir_e2e = self.cbf_basis_tint(main_graph["distance"], cosφ_cab)
        bases_e2e = {
            "rad": self.mlp_rbf_tint(basis_rad_main_raw),
            "cir": self.mlp_cbf_tint(
                rad_basis=rad_cir_e2e,
                sph_basis=cir_e2e,
                idx_sph_outer=trip_idx_e2e["out"],
                idx_sph_inner=trip_idx_e2e["out_agg"],
            ),
        }

        # Quadruplet interaction (optional).
        bases_qint: dict[str, torch.Tensor] = {}
        if self.quad_interaction:
            cosφ_cab_q, cosφ_abd, angle_cabd = self._calculate_quad_angles(
                main_graph["vector"], qint_graph["vector"], quad_idx
            )
            rad_cir_q, cir_q = self.cbf_basis_qint(qint_graph["distance"], cosφ_abd)
            rad_sph_q, sph_q = self.sbf_basis_qint(
                main_graph["distance"], cosφ_cab_q[quad_idx["trip_out_to_quad"]], angle_cabd
            )
            bases_qint = {
                "rad": self.mlp_rbf_qint(basis_rad_main_raw),
                "cir": self.mlp_cbf_qint(
                    rad_basis=rad_cir_q,
                    sph_basis=cir_q,
                    idx_sph_outer=quad_idx["triplet_in"]["out"],
                ),
                "sph": self.mlp_sbf_qint(
                    rad_basis=rad_sph_q,
                    sph_basis=sph_q,
                    idx_sph_outer=quad_idx["out"],
                    idx_sph_inner=quad_idx["out_agg"],
                ),
            }

        # Atom -> edge (mixed triplet) interaction (optional).
        bases_a2e: dict[str, torch.Tensor] = {}
        if self.atom_edge_interaction:
            rad_a2ee2a = self.radial_basis_aeaint(a2ee2a_graph["distance"])
            cosφ_cab_a2e = inner_product_clamped(
                main_graph["vector"][trip_idx_a2e["out"]],
                a2ee2a_graph["vector"][trip_idx_a2e["in"]],
            )
            rad_cir_a2e, cir_a2e = self.cbf_basis_aeint(main_graph["distance"], cosφ_cab_a2e)
            bases_a2e = {
                "rad": self.mlp_rbf_aeint(rad_a2ee2a),
                "cir": self.mlp_cbf_aeint(
                    rad_basis=rad_cir_a2e,
                    sph_basis=cir_a2e,
                    idx_sph_outer=trip_idx_a2e["out"],
                    idx_sph_inner=trip_idx_a2e["out_agg"],
                ),
            }

        # Edge -> atom (mixed triplet) interaction (optional).
        bases_e2a: dict[str, torch.Tensor] = {}
        if self.edge_atom_interaction:
            cosφ_cab_e2a = inner_product_clamped(
                a2ee2a_graph["vector"][trip_idx_e2a["out"]],
                main_graph["vector"][trip_idx_e2a["in"]],
            )
            rad_cir_e2a, cir_e2a = self.cbf_basis_eaint(a2ee2a_graph["distance"], cosφ_cab_e2a)
            bases_e2a = {
                "rad": self.mlp_rbf_eaint(basis_rad_main_raw),
                "cir": self.mlp_cbf_eaint(
                    rad_basis=rad_cir_e2a,
                    sph_basis=cir_e2a,
                    idx_rad_outer=a2ee2a_graph["edge_index"][1],
                    idx_rad_inner=a2ee2a_graph["target_neighbor_idx"],
                    idx_sph_outer=trip_idx_e2a["out"],
                    idx_sph_inner=trip_idx_e2a["out_agg"],
                    num_atoms=num_atoms,
                ),
            }

        # Atom -> atom interaction (optional).
        basis_a2a_rad = None
        if self.atom_interaction:
            rad_a2a = self.radial_basis_aint(a2a_graph["distance"])
            basis_a2a_rad = self.mlp_rbf_aint(
                rad_basis=rad_a2a,
                idx_rad_outer=a2a_graph["edge_index"][1],
                idx_rad_inner=a2a_graph["target_neighbor_idx"],
                num_atoms=num_atoms,
            )

        return (
            basis_rad_main_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
        id_swap: torch.Tensor,
        id3_expand_ba: torch.Tensor,
        id3_reduce_ca: torch.Tensor,
        Kidx3: torch.Tensor,
        # Quadruplet (only when quad_interaction=True):
        qint_edge_index: torch.Tensor | None = None,
        qint_edge_weight: torch.Tensor | None = None,
        qint_edge_vec: torch.Tensor | None = None,
        id4_expand_intm_db: torch.Tensor | None = None,
        id4_expand_intm_ab: torch.Tensor | None = None,
        id4_reduce_intm_ab: torch.Tensor | None = None,
        id4_reduce_intm_ca: torch.Tensor | None = None,
        id4_reduce_ca: torch.Tensor | None = None,
        id4_expand_abd: torch.Tensor | None = None,
        id4_reduce_cab: torch.Tensor | None = None,
        Kidx4: torch.Tensor | None = None,
        # a2ee2a graph (needed when atom_edge_interaction or edge_atom_interaction):
        a2ee2a_edge_index: torch.Tensor | None = None,
        a2ee2a_edge_weight: torch.Tensor | None = None,
        a2ee2a_edge_vec: torch.Tensor | None = None,
        # Mixed-triplet a2e (atom_edge_interaction=True):
        trip_a2e_in: torch.Tensor | None = None,
        trip_a2e_out: torch.Tensor | None = None,
        trip_a2e_out_agg: torch.Tensor | None = None,
        # Mixed-triplet e2a (edge_atom_interaction=True):
        trip_e2a_in: torch.Tensor | None = None,
        trip_e2a_out: torch.Tensor | None = None,
        trip_e2a_out_agg: torch.Tensor | None = None,
        # a2a graph (atom_interaction=True):
        a2a_edge_index: torch.Tensor | None = None,
        a2a_edge_weight: torch.Tensor | None = None,
        a2a_edge_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a full forward pass and return per-atom output spectra.

        Args:
            z: Atomic numbers, shape ``(nAtoms,)``.
            edge_index: Main edge index, shape ``(2, nEdges)``.
            edge_weight: Edge distances in **angstrom**, shape ``(nEdges,)``.
            edge_vec: PBC-aware edge displacement vectors, shape
                ``(nEdges, 3)``.
            id_swap: Index of the reverse edge for each edge, shape ``(nEdges,)``.
            id3_expand_ba: E2E triplet expansion index, shape ``(nTriplets,)``.
            id3_reduce_ca: E2E triplet reduction index, shape ``(nTriplets,)``.
            Kidx3: Per-edge triplet enumeration for E2E, shape ``(nTriplets,)``.
            qint_edge_index: Quadruplet graph edge index (optional).
            qint_edge_weight: Quadruplet edge distances (optional).
            qint_edge_vec: Quadruplet PBC-aware edge displacement vectors
                (optional).
            id4_expand_intm_db: Quadruplet index (optional).
            id4_expand_intm_ab: Quadruplet index (optional).
            id4_reduce_intm_ab: Quadruplet index (optional).
            id4_reduce_intm_ca: Quadruplet index (optional).
            id4_reduce_ca: Quadruplet reduction index (optional).
            id4_expand_abd: Quadruplet dihedral expansion index (optional).
            id4_reduce_cab: Quadruplet dihedral reduction index (optional).
            Kidx4: Per-edge quadruplet enumeration (optional).
            a2ee2a_edge_index: A2EE2A graph edge index (optional).
            a2ee2a_edge_weight: A2EE2A edge distances (optional).
            a2ee2a_edge_vec: A2EE2A edge displacement vectors (optional).
            trip_a2e_in: A2E triplet input index (optional).
            trip_a2e_out: A2E triplet output index (optional).
            trip_a2e_out_agg: A2E triplet per-edge enumeration (optional).
            trip_e2a_in: E2A triplet input index (optional).
            trip_e2a_out: E2A triplet output index (optional).
            trip_e2a_out_agg: E2A triplet per-edge enumeration (optional).
            a2a_edge_index: A2A graph edge index (optional).
            a2a_edge_weight: A2A edge distances (optional).
            a2a_edge_vec: A2A edge displacement vectors (optional; retained
                for graph-structure parity, not used by the current atom-pair
                basis path).

        Returns:
            Per-atom output spectra of shape ``(nAtoms, num_targets)``.
        """
        z = z.long()
        num_atoms = z.size(0)

        # Group the flat forward-arg tensors into the nested dicts consumed by
        # ``_get_bases`` and the interaction blocks. The only non-trivial work
        # is ``target_neighbor_idx`` (per-edge index into the destination
        # atom's neighbor list), required by basis-embedding scatters inside
        # ``_get_bases``.
        main_graph = {"edge_index": edge_index, "distance": edge_weight, "vector": edge_vec}
        trip_idx_e2e = {"in": id3_expand_ba, "out": id3_reduce_ca, "out_agg": Kidx3}

        qint_graph: dict[str, torch.Tensor] = {}
        quad_idx: dict = {}
        if self.quad_interaction:
            assert qint_edge_index is not None
            assert qint_edge_weight is not None
            assert qint_edge_vec is not None
            assert id4_expand_intm_db is not None
            assert id4_expand_intm_ab is not None
            assert id4_reduce_intm_ab is not None
            assert id4_reduce_intm_ca is not None
            assert id4_reduce_ca is not None
            assert id4_expand_abd is not None
            assert id4_reduce_cab is not None
            assert Kidx4 is not None
            qint_graph = {"edge_index": qint_edge_index, "distance": qint_edge_weight, "vector": qint_edge_vec}
            quad_idx = {
                "triplet_in": {"in": id4_expand_intm_db, "out": id4_expand_intm_ab},
                "triplet_out": {"in": id4_reduce_intm_ab, "out": id4_reduce_intm_ca},
                "out": id4_reduce_ca,
                "trip_in_to_quad": id4_expand_abd,
                "trip_out_to_quad": id4_reduce_cab,
                "out_agg": Kidx4,
            }

        a2ee2a_graph: dict[str, torch.Tensor] = {}
        trip_idx_a2e: dict[str, torch.Tensor] = {}
        trip_idx_e2a: dict[str, torch.Tensor] = {}
        if self.atom_edge_interaction or self.edge_atom_interaction:
            assert a2ee2a_edge_index is not None
            assert a2ee2a_edge_weight is not None
            assert a2ee2a_edge_vec is not None
            a2ee2a_graph = {
                "edge_index": a2ee2a_edge_index,
                "distance": a2ee2a_edge_weight,
                "vector": a2ee2a_edge_vec,
                "target_neighbor_idx": get_inner_idx(a2ee2a_edge_index[1], dim_size=num_atoms),
            }
        if self.atom_edge_interaction:
            assert trip_a2e_in is not None
            assert trip_a2e_out is not None
            assert trip_a2e_out_agg is not None
            trip_idx_a2e = {"in": trip_a2e_in, "out": trip_a2e_out, "out_agg": trip_a2e_out_agg}
        if self.edge_atom_interaction:
            assert trip_e2a_in is not None
            assert trip_e2a_out is not None
            assert trip_e2a_out_agg is not None
            trip_idx_e2a = {"in": trip_e2a_in, "out": trip_e2a_out, "out_agg": trip_e2a_out_agg}

        a2a_graph: dict[str, torch.Tensor] = {}
        if self.atom_interaction:
            assert a2a_edge_index is not None
            assert a2a_edge_weight is not None
            assert a2a_edge_vec is not None
            a2a_graph = {
                "edge_index": a2a_edge_index,
                "distance": a2a_edge_weight,
                "vector": a2a_edge_vec,
                "target_neighbor_idx": get_inner_idx(a2a_edge_index[1], dim_size=num_atoms),
            }

        _, idx_t = main_graph["edge_index"]

        (
            basis_rad_main_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        ) = self._get_bases(
            main_graph=main_graph,
            a2a_graph=a2a_graph,
            a2ee2a_graph=a2ee2a_graph,
            qint_graph=qint_graph,
            trip_idx_e2e=trip_idx_e2e,
            trip_idx_a2e=trip_idx_a2e,
            trip_idx_e2a=trip_idx_e2a,
            quad_idx=quad_idx,
            num_atoms=num_atoms,
        )

        # Embedding
        h = self.atom_emb(z)
        m = self.edge_emb(h, basis_rad_main_raw, main_graph["edge_index"])

        x_E = self.out_blocks[0](h, m, basis_output, idx_t)
        xs_E = [x_E]

        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                bases_qint=bases_qint,
                bases_e2e=bases_e2e,
                bases_a2e=bases_a2e,
                bases_e2a=bases_e2a,
                basis_a2a_rad=basis_a2a_rad,
                basis_atom_update=basis_atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
            )
            x_E = self.out_blocks[i + 1](h, m, basis_output, idx_t)
            xs_E.append(x_E)

        # Global per-atom output
        x_E = self.out_mlp_E(torch.cat(xs_E, dim=-1))
        with torch.autocast("cuda", enabled=False):
            out = self.out_energy(x_E.float())
        return out  # (nAtoms, num_targets)

    @property
    def num_params(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """No-op: GemNet-OC uses its own He-orthogonal init scheme.

        Args:
            weights_init: Unused generic weight-initialiser selector.
            bias_init: Unused generic bias-initialiser selector.
            **kwargs: Ignored extra initialisation options.

        Note:
            GemNet-OC layer constructors call ``reset_parameters`` internally;
            the generic ``weights_init`` and ``bias_init`` selectors are
            ignored.
        """
        logging.warning("GemNetOC uses custom weight initialization; weights_init and bias_init arguments are ignored.")

    @property
    def signature(self):
        """Return the model signature as a :class:`~xanesnet.serialization.config.Config`.

        Returns:
            A config object containing the constructor hyperparameters needed
            to recreate the model.

        Note:
            ``scale_file`` is always set to ``None`` in the signature. The fitted
            scale-factor values are carried by the model ``state_dict`` as
            non-trainable parameters, making saved checkpoints fully portable.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "num_targets": self.num_targets,
                "num_spherical": self.num_spherical,
                "num_radial": self.num_radial,
                "num_blocks": self.num_blocks,
                "emb_size_atom": self.emb_size_atom,
                "emb_size_edge": self.emb_size_edge,
                "emb_size_trip_in": self.emb_size_trip_in,
                "emb_size_trip_out": self.emb_size_trip_out,
                "emb_size_quad_in": self.emb_size_quad_in,
                "emb_size_quad_out": self.emb_size_quad_out,
                "emb_size_aint_in": self.emb_size_aint_in,
                "emb_size_aint_out": self.emb_size_aint_out,
                "emb_size_rbf": self.emb_size_rbf,
                "emb_size_cbf": self.emb_size_cbf,
                "emb_size_sbf": self.emb_size_sbf,
                "num_before_skip": self.num_before_skip,
                "num_after_skip": self.num_after_skip,
                "num_concat": self.num_concat,
                "num_atom": self.num_atom,
                "num_output_afteratom": self.num_output_afteratom,
                "num_atom_emb_layers": self.num_atom_emb_layers,
                "num_global_out_layers": self.num_global_out_layers,
                "cutoff": self.cutoff,
                "cutoff_qint": self.cutoff_qint,
                "cutoff_aeaint": self.cutoff_aeaint,
                "cutoff_aint": self.cutoff_aint,
                "rbf": self.rbf_cfg.as_dict(),
                "rbf_spherical": self.rbf_spherical_cfg.as_dict(),
                "envelope": self.envelope_cfg.as_dict(),
                "cbf": self.cbf_cfg.as_dict(),
                "sbf": self.sbf_cfg.as_dict(),
                "output_init": self.output_init,
                "activation": self.activation,
                "quad_interaction": self.quad_interaction,
                "atom_edge_interaction": self.atom_edge_interaction,
                "edge_atom_interaction": self.edge_atom_interaction,
                "atom_interaction": self.atom_interaction,
                "scale_basis": self.scale_basis,
                "num_elements": self.num_elements,
                # scale_file is intentionally None in the signature: the actual fitted
                # values are carried by state_dict (ScaleFactor parameters), making
                # the saved model fully self-contained and portable across machines.
                "scale_file": None,
            }
        )
        return signature
