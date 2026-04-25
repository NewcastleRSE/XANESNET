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
    """
    GemNet-OC adapted for per-atom XANES spectrum prediction.
    Ported from the fairchem-core reference (MIT License).

    Periodicity handling
    --------------------
    This model is *agnostic* to periodicity. It consumes precomputed,
    PBC-aware tensors (``edge_vec``, ``edge_weight``, and the optional
    ``qint_*``/``a2ee2a_*``/``a2a_*`` counterparts) produced by
    :class:`~xanesnet.datasets.torchgeometric.gemnet.GemNetDataset`. The dataset
    builds each graph with ``build_edges`` on a ``pymatgen`` object — which
    handles periodic (``Structure``) and non-periodic (``Molecule``) inputs
    uniformly and emits lattice-corrected vectors for periodic self-image
    edges. The model never reads raw ``batch.pos`` and never recomputes
    distances / vectors, so periodic and non-periodic structures travel the
    same code path here without any special-casing.

    Basis-function configuration
    ----------------------------
    ``rbf``, ``rbf_spherical``, ``envelope``, ``cbf`` and ``sbf`` are
    :class:`~xanesnet.serialization.config.Config` objects (the form returned
    by ``Config.as_kwargs`` for nested YAML sub-sections). Each carries a
    required ``name`` key plus optional hyperparameters, e.g.
    ``Config({'name': 'polynomial', 'exponent': 5})``. Programmatic callers
    (tests, notebooks) must wrap raw dicts in ``Config(...)`` before passing
    them in.
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

    def _set_cutoffs(self, cutoff, cutoff_qint, cutoff_aeaint, cutoff_aint) -> None:
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
        self, num_radial, num_spherical, rbf, rbf_spherical, envelope, cbf, sbf, scale_basis
    ) -> None:
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

    def _init_shared_basis_layers(self, num_radial, num_spherical, emb_size_rbf, emb_size_cbf, emb_size_sbf) -> None:
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

    def _calculate_quad_angles(self, V_st, V_qint_st, quad_idx):
        """
        Circular/spherical angle tensors for quadruplet-based MP.
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
        main_graph,
        a2a_graph,
        a2ee2a_graph,
        qint_graph,
        trip_idx_e2e,
        trip_idx_a2e,
        trip_idx_e2a,
        quad_idx,
        num_atoms,
    ):
        """
        Compute all radial / circular / spherical basis tensors used by the
        interaction and output blocks.

        Structure: one branch per interaction type. Each branch computes the
        raw basis tensors (radial + angles) and immediately applies the
        corresponding ``mlp_*`` / ``BasisEmbedding`` layers, returning a small
        dict of embedded bases. Branches disabled by the interaction flags
        contribute an empty dict / ``None`` (the downstream ``InteractionBlock``
        checks those flags consistently).
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
        return sum(p.numel() for p in self.parameters())

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """
        GemNet-OC uses its own HE-orthogonal / uniform init scheme internally
        (layer constructors call ``reset_parameters``);
        the generic ``weights_init`` / ``bias_init`` selectors are therefore ignored.
        """
        logging.warning("GemNetOC uses custom weight initialization; weights_init and bias_init arguments are ignored.")

    @property
    def signature(self):
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
                # ``scale_file`` is intentionally None in the signature: the
                # actual fitted values are carried by ``state_dict``
                # (ScaleFactor parameters), making the saved model fully
                # self-contained and portable across machines.
                "scale_file": None,
            }
        )
        return signature
