# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either Version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

"""GemNet and GemNet-OC PyTorch Geometric dataset implementation."""

import logging
from typing import Any, Protocol

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config
from xanesnet.utils.graph import build_edges
from xanesnet.utils.graph.gemnet_indices import (
    compute_id_swap,
    compute_mixed_triplets,
    compute_quadruplets,
    compute_triplets,
)

from ..base import SavePathFn, TorchGeometricDataset
from ..registry import DatasetRegistry

SPECTRUM_KEYS = ["XANES", "XANES_K"]  # TODO maybe put this somewhere more central?


class GemNetData(Data):
    """PyG ``Data`` subclass for GemNet and GemNet-OC samples.

    The custom ``__inc__`` is critical for correct PyG batching of all edge /
    triplet / quadruplet / intermediate indices across multiple graphs.
    """

    # Node-level indices (offset by num_nodes when batched)
    _NODE_KEYS = {"id_c", "id_a", "id4_int_b", "id4_int_a"}
    # Main-graph edge-level indices (offset by num_main_edges)
    _MAIN_EDGE_KEYS = {
        "id_swap",
        "id3_expand_ba",
        "id3_reduce_ca",
        "id4_reduce_ca",
        "id4_expand_db",
        "id4_reduce_intm_ca",
        "id4_expand_intm_db",
        # OC mixed triplets where "out" edges are from the main graph
        "trip_e2e_in",
        "trip_e2e_out",
        "trip_a2e_out",
        "trip_e2a_in",
    }
    # Interaction-graph edge-level (offset by num_int_edges)
    _INT_EDGE_KEYS = {
        "id4_reduce_intm_ab",
        "id4_expand_intm_ab",
    }
    # a2ee2a-graph edge-level (offset by num_a2ee2a_edges)
    _A2EE2A_EDGE_KEYS = {
        "trip_a2e_in",
        "trip_e2a_out",
    }
    # a2a-graph edge-level
    _A2A_EDGE_KEYS: set[str] = set()
    # qint-graph edge-level
    _QINT_EDGE_KEYS: set[str] = set()
    # Intermediate-ca level (offset by num_intm_ca)
    _INTM_CA_KEYS = {"id4_reduce_cab"}
    # Intermediate-db level (offset by num_intm_db)
    _INTM_DB_KEYS = {"id4_expand_abd"}
    # Ragged inner indices (no offset)
    _NO_INC_KEYS = {"Kidx3", "Kidx4", "trip_e2e_out_agg", "trip_a2e_out_agg", "trip_e2a_out_agg"}

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Return PyG batching increments for GemNet index fields.

        Args:
            key: Data attribute currently being batched.
            value: Attribute value currently being batched.
            *args: Additional PyG arguments.
            **kwargs: Additional PyG keyword arguments.

        Returns:
            Increment used by PyG for ``key``.
        """
        if (
            key in {"edge_index", "int_edge_index", "a2ee2a_edge_index", "a2a_edge_index", "qint_edge_index"}
            or key in self._NODE_KEYS
        ):
            return self.num_nodes
        if key in self._MAIN_EDGE_KEYS:
            ei = self.edge_index
            return ei.size(1) if ei is not None else 0
        if key in self._INT_EDGE_KEYS:
            # Interaction edge graph is a 2xE tensor or an index into it
            return self.int_edge_index.size(1) if getattr(self, "int_edge_index", None) is not None else 0
        if key in self._A2EE2A_EDGE_KEYS:
            a2ee2a = getattr(self, "a2ee2a_edge_index", None)
            return a2ee2a.size(1) if a2ee2a is not None else 0
        if key in self._A2A_EDGE_KEYS:
            a2a = getattr(self, "a2a_edge_index", None)
            return a2a.size(1) if a2a is not None else 0
        if key in self._QINT_EDGE_KEYS:
            q = getattr(self, "qint_edge_index", None)
            return q.size(1) if q is not None else 0
        if key in self._INTM_CA_KEYS:
            v = getattr(self, "id4_reduce_intm_ca", None)
            return v.size(0) if v is not None else 0
        if key in self._INTM_DB_KEYS:
            v = getattr(self, "id4_expand_intm_db", None)
            return v.size(0) if v is not None else 0
        if key in self._NO_INC_KEYS:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the concatenation dimension for GemNet graph fields.

        Args:
            key: Data attribute currently being batched.
            value: Attribute value currently being batched.
            *args: Additional PyG arguments.
            **kwargs: Additional PyG keyword arguments.

        Returns:
            Concatenation dimension used by PyG for ``key``.
        """
        if key in {"edge_index", "int_edge_index", "a2ee2a_edge_index", "a2a_edge_index", "qint_edge_index"}:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class GemNetBatch(Protocol):
    """Protocol for batches emitted by ``GemNetDataset.collate_fn``."""

    x: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_vec: torch.Tensor
    id_c: torch.Tensor
    id_a: torch.Tensor
    id_swap: torch.Tensor
    id3_reduce_ca: torch.Tensor
    id3_expand_ba: torch.Tensor
    Kidx3: torch.Tensor
    absorber_mask: torch.Tensor
    energies: torch.Tensor
    intensities: torch.Tensor
    file_name: list[str]


@DatasetRegistry.register("gemnet")
@DatasetRegistry.register("gemnet_oc")
class GemNetDataset(TorchGeometricDataset):
    """Dataset for GemNet and GemNet-OC graph inputs.

    Mixing graph methods is supported (e.g. ``graph_method="voronoi"`` for a
    compact main graph paired with ``int_graph_method="radius"`` for a larger
    interaction graph). A derived graph is reused from a previously-built one
    only when *all* of ``(cutoff, max_num_neighbors, method, min_facet_area,
    cov_radii_scale)`` match; otherwise it is built from scratch.

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw datasource of pymatgen structures or molecules.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        cutoff: Main graph cutoff in **Angstrom**.
        max_num_neighbors: Main graph per-source neighbor cap.
        graph_method: Main graph construction method.
        min_facet_area: Optional Voronoi facet-area threshold for the main graph.
        cov_radii_scale: Covalent-radii scale for the main graph.
        quadruplets: Whether to compute quadruplet indices.
        int_cutoff: Interaction graph cutoff in **Angstrom**.
        int_max_neighbors: Optional interaction graph per-source neighbor cap.
        int_graph_method: Optional interaction graph construction method.
        int_min_facet_area: Optional interaction Voronoi facet-area threshold.
        int_cov_radii_scale: Optional interaction covalent-radii scale.
        oc_mode: Whether to precompute GemNet-OC auxiliary graphs and mixed triplets.
        oc_cutoff_aeaint: Atom-edge-atom graph cutoff in **Angstrom**.
        oc_cutoff_aint: Atom-atom graph cutoff in **Angstrom**.
        oc_max_neighbors_aeaint: Atom-edge-atom graph per-source neighbor cap.
        oc_max_neighbors_aint: Atom-atom graph per-source neighbor cap.
        oc_graph_method_aeaint: Atom-edge-atom graph construction method override.
        oc_min_facet_area_aeaint: Atom-edge-atom Voronoi facet-area threshold.
        oc_cov_radii_scale_aeaint: Atom-edge-atom covalent-radii scale.
        oc_graph_method_aint: Atom-atom graph construction method override.
        oc_min_facet_area_aint: Atom-atom Voronoi facet-area threshold.
        oc_cov_radii_scale_aint: Atom-atom covalent-radii scale.
    """

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
        skip_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
        # params:
        cutoff: float,
        max_num_neighbors: int,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
        quadruplets: bool,
        int_cutoff: float | None,
        int_max_neighbors: int | None = None,
        int_graph_method: str | None = None,
        int_min_facet_area: float | str | None = None,
        int_cov_radii_scale: float | None = None,
        oc_mode: bool = False,
        oc_cutoff_aeaint: float | None = None,
        oc_cutoff_aint: float | None = None,
        oc_max_neighbors_aeaint: int | None = None,
        oc_max_neighbors_aint: int | None = None,
        oc_graph_method_aeaint: str | None = None,
        oc_min_facet_area_aeaint: float | str | None = None,
        oc_cov_radii_scale_aeaint: float | None = None,
        oc_graph_method_aint: str | None = None,
        oc_min_facet_area_aint: float | str | None = None,
        oc_cov_radii_scale_aint: float | None = None,
    ) -> None:
        """Initialize the GemNet dataset."""
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.graph_method = graph_method
        self.min_facet_area = min_facet_area
        self.cov_radii_scale = cov_radii_scale
        self.quadruplets = quadruplets
        self.int_cutoff = int_cutoff if int_cutoff is not None else cutoff
        self.int_max_neighbors = int_max_neighbors if int_max_neighbors is not None else max_num_neighbors
        self.int_graph_method = int_graph_method if int_graph_method is not None else graph_method
        self.int_min_facet_area = int_min_facet_area if int_min_facet_area is not None else min_facet_area
        self.int_cov_radii_scale = int_cov_radii_scale if int_cov_radii_scale is not None else cov_radii_scale
        self.oc_mode = oc_mode
        self.oc_cutoff_aeaint = oc_cutoff_aeaint if oc_cutoff_aeaint is not None else cutoff
        self.oc_cutoff_aint = (
            oc_cutoff_aint if oc_cutoff_aint is not None else max(self.cutoff, self.oc_cutoff_aeaint, self.int_cutoff)
        )
        self.oc_max_neighbors_aeaint = (
            oc_max_neighbors_aeaint if oc_max_neighbors_aeaint is not None else max_num_neighbors
        )
        self.oc_max_neighbors_aint = oc_max_neighbors_aint if oc_max_neighbors_aint is not None else max_num_neighbors
        self.oc_graph_method_aeaint = oc_graph_method_aeaint if oc_graph_method_aeaint is not None else graph_method
        self.oc_min_facet_area_aeaint = (
            oc_min_facet_area_aeaint if oc_min_facet_area_aeaint is not None else min_facet_area
        )
        self.oc_cov_radii_scale_aeaint = (
            oc_cov_radii_scale_aeaint if oc_cov_radii_scale_aeaint is not None else cov_radii_scale
        )
        self.oc_graph_method_aint = oc_graph_method_aint if oc_graph_method_aint is not None else graph_method
        self.oc_min_facet_area_aint = oc_min_facet_area_aint if oc_min_facet_area_aint is not None else min_facet_area
        self.oc_cov_radii_scale_aint = (
            oc_cov_radii_scale_aint if oc_cov_radii_scale_aint is not None else cov_radii_scale
        )

        if oc_mode and not quadruplets:
            # GemNet-OC always needs quadruplet indices for its standard config;
            # allow disabling only if the user explicitly sets quadruplets=False.
            logging.info(
                "GemNetDataset: oc_mode=True without quadruplets=True. Quadruplet "
                "indices will NOT be computed; GemNet-OC's quad_interaction must be False."
            )

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into one GemNet graph sample.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps sample sequence numbers to output paths.

        Returns:
            ``1`` when the graph was saved, otherwise ``0`` when skipped.
        """
        pmg_obj = self.datasource[idx]
        for key in SPECTRUM_KEYS:
            if key in pmg_obj.site_properties.keys():
                break
        else:
            logging.warning(
                f"No XANES spectrum found for sample {idx} ({pmg_obj.properties.get('file_name', '')}); skipping."
            )
            return 0

        xanes = np.array(pmg_obj.site_properties[key], dtype=object)
        xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()  # noqa: E711
        if len(xanes_idxs) == 0:
            logging.warning(f"No absorbers for sample {idx}; skipping.")
            return 0

        xanes = xanes[xanes_idxs]
        intensities = np.stack([x["intensities"] for x in xanes]).astype(np.float32)
        energies = np.stack([x["energies"] for x in xanes]).astype(np.float32)

        n_atoms = len(pmg_obj.atomic_numbers)
        absorber_mask = torch.zeros(n_atoms, dtype=torch.bool)
        absorber_mask[xanes_idxs] = True

        atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
        cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)

        main_params = (
            self.cutoff,
            self.max_num_neighbors,
            self.graph_method,
            self.min_facet_area,
            self.cov_radii_scale,
        )
        edge_index, edge_weight, edge_vec, _ = build_edges(
            pmg_obj,
            self.cutoff,
            self.max_num_neighbors,
            compute_vectors=True,
            method=self.graph_method,
            min_facet_area=self.min_facet_area,
            cov_radii_scale=self.cov_radii_scale,
        )
        assert edge_vec is not None

        if edge_index.size(1) > 0:
            id_swap = compute_id_swap(edge_index, edge_vec)
            id3_reduce_ca, id3_expand_ba, Kidx3 = compute_triplets(edge_index, n_atoms)
        else:
            id_swap = torch.empty(0, dtype=torch.int64)
            id3_reduce_ca = torch.empty(0, dtype=torch.int64)
            id3_expand_ba = torch.empty(0, dtype=torch.int64)
            Kidx3 = torch.empty(0, dtype=torch.int64)

        data_fields: dict[str, Any] = {
            "x": atomic_numbers,
            "pos": cart_coords,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "edge_vec": edge_vec,
            "id_c": edge_index[0].clone().to(torch.int64),
            "id_a": edge_index[1].clone().to(torch.int64),
            "id_swap": id_swap,
            "id3_reduce_ca": id3_reduce_ca,
            "id3_expand_ba": id3_expand_ba,
            "Kidx3": Kidx3,
            "energies": torch.tensor(energies, dtype=torch.float32),
            "intensities": torch.tensor(intensities, dtype=torch.float32),
            "absorber_mask": absorber_mask,
            "file_name": pmg_obj.properties["file_name"],
        }

        int_params = (
            self.int_cutoff,
            self.int_max_neighbors,
            self.int_graph_method,
            self.int_min_facet_area,
            self.int_cov_radii_scale,
        )
        int_edge_index = int_edge_weight = int_edge_vec = None
        if self.quadruplets:
            if int_params == main_params:
                int_edge_index = edge_index
                int_edge_weight = edge_weight
                int_edge_vec = edge_vec
            else:
                int_edge_index, int_edge_weight, int_edge_vec, _ = build_edges(
                    pmg_obj,
                    self.int_cutoff,
                    self.int_max_neighbors,
                    compute_vectors=True,
                    method=self.int_graph_method,
                    min_facet_area=self.int_min_facet_area,
                    cov_radii_scale=self.int_cov_radii_scale,
                )
            assert int_edge_vec is not None
            data_fields.update(
                {
                    "int_edge_index": int_edge_index,
                    "int_edge_weight": int_edge_weight,
                    "int_edge_vec": int_edge_vec,
                    "id4_int_b": int_edge_index[0].clone().to(torch.int64),
                    "id4_int_a": int_edge_index[1].clone().to(torch.int64),
                }
            )
            if int_edge_index.size(1) > 0 and edge_index.size(1) > 0:
                quad = compute_quadruplets(edge_index, edge_vec, int_edge_index, int_edge_vec, n_atoms)
            else:
                empty = torch.empty(0, dtype=torch.int64)
                quad = dict(
                    id4_reduce_ca=empty,
                    id4_expand_db=empty,
                    id4_reduce_cab=empty,
                    id4_expand_abd=empty,
                    id4_reduce_intm_ca=empty,
                    id4_expand_intm_db=empty,
                    id4_reduce_intm_ab=empty,
                    id4_expand_intm_ab=empty,
                    Kidx4=empty,
                )
            data_fields.update(quad)

        if self.oc_mode:
            aeaint_params = (
                self.oc_cutoff_aeaint,
                self.oc_max_neighbors_aeaint,
                self.oc_graph_method_aeaint,
                self.oc_min_facet_area_aeaint,
                self.oc_cov_radii_scale_aeaint,
            )
            if aeaint_params == main_params:
                a2ee2a_edge_index = edge_index
                a2ee2a_edge_weight = edge_weight
                a2ee2a_edge_vec = edge_vec
            else:
                a2ee2a_edge_index, a2ee2a_edge_weight, a2ee2a_edge_vec, _ = build_edges(
                    pmg_obj,
                    self.oc_cutoff_aeaint,
                    self.oc_max_neighbors_aeaint,
                    compute_vectors=True,
                    method=self.oc_graph_method_aeaint,
                    min_facet_area=self.oc_min_facet_area_aeaint,
                    cov_radii_scale=self.oc_cov_radii_scale_aeaint,
                )
            aint_params = (
                self.oc_cutoff_aint,
                self.oc_max_neighbors_aint,
                self.oc_graph_method_aint,
                self.oc_min_facet_area_aint,
                self.oc_cov_radii_scale_aint,
            )
            if self.quadruplets and aint_params == int_params:
                a2a_edge_index = int_edge_index
                a2a_edge_weight = int_edge_weight
                a2a_edge_vec = int_edge_vec
            elif aint_params == main_params:
                a2a_edge_index = edge_index
                a2a_edge_weight = edge_weight
                a2a_edge_vec = edge_vec
            else:
                a2a_edge_index, a2a_edge_weight, a2a_edge_vec, _ = build_edges(
                    pmg_obj,
                    self.oc_cutoff_aint,
                    self.oc_max_neighbors_aint,
                    compute_vectors=True,
                    method=self.oc_graph_method_aint,
                    min_facet_area=self.oc_min_facet_area_aint,
                    cov_radii_scale=self.oc_cov_radii_scale_aint,
                )
            assert a2ee2a_edge_vec is not None and a2a_edge_vec is not None

            data_fields.update(
                {
                    "a2ee2a_edge_index": a2ee2a_edge_index,
                    "a2ee2a_edge_weight": a2ee2a_edge_weight,
                    "a2ee2a_edge_vec": a2ee2a_edge_vec,
                    "a2a_edge_index": a2a_edge_index,
                    "a2a_edge_weight": a2a_edge_weight,
                    "a2a_edge_vec": a2a_edge_vec,
                }
            )

            if self.quadruplets:
                data_fields["qint_edge_index"] = data_fields["int_edge_index"]
                data_fields["qint_edge_weight"] = data_fields["int_edge_weight"]
                data_fields["qint_edge_vec"] = data_fields["int_edge_vec"]
            else:
                qint_edge_index, qint_edge_weight, qint_edge_vec, _ = build_edges(
                    pmg_obj,
                    self.int_cutoff,
                    self.int_max_neighbors,
                    compute_vectors=True,
                    method=self.int_graph_method,
                    min_facet_area=self.int_min_facet_area,
                    cov_radii_scale=self.int_cov_radii_scale,
                )
                assert qint_edge_vec is not None
                data_fields["qint_edge_index"] = qint_edge_index
                data_fields["qint_edge_weight"] = qint_edge_weight
                data_fields["qint_edge_vec"] = qint_edge_vec

            data_fields["trip_e2e_in"] = data_fields["id3_expand_ba"]
            data_fields["trip_e2e_out"] = data_fields["id3_reduce_ca"]
            data_fields["trip_e2e_out_agg"] = data_fields["Kidx3"]

            a2e = compute_mixed_triplets(
                main_edge_index=edge_index,
                main_edge_vec=edge_vec,
                other_edge_index=a2ee2a_edge_index,
                other_edge_vec=a2ee2a_edge_vec,
                num_nodes=n_atoms,
                to_outedge=False,
            )
            e2a = compute_mixed_triplets(
                main_edge_index=a2ee2a_edge_index,
                main_edge_vec=a2ee2a_edge_vec,
                other_edge_index=edge_index,
                other_edge_vec=edge_vec,
                num_nodes=n_atoms,
                to_outedge=False,
            )
            data_fields.update(
                {
                    "trip_a2e_in": a2e["in_"],
                    "trip_a2e_out": a2e["out"],
                    "trip_a2e_out_agg": a2e["out_agg"],
                    "trip_e2a_in": e2a["in_"],
                    "trip_e2a_out": e2a["out"],
                    "trip_e2a_out_agg": e2a["out_agg"],
                }
            )

        struct = GemNetData(**data_fields)
        self._save_data(struct, save_path_fn(0))
        return 1

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        """Collate GemNet graph samples into one PyG batch.

        Args:
            batch: GemNet graph samples loaded by ``__getitem__``.

        Returns:
            PyG batch with target tensors concatenated over absorber sites.
        """
        fields_to_cat = ["energies", "intensities", "absorber_mask"]
        batched = Batch.from_data_list(batch, exclude_keys=fields_to_cat)
        for field in fields_to_cat:
            setattr(batched, field, torch.cat([getattr(d, field) for d in batch], dim=0))
        return batched

    @staticmethod
    def _save_data(data: GemNetData, path: str) -> None:
        """Save one GemNet data object as a tensor dictionary.

        Args:
            data: Data object to serialize.
            path: Destination ``.pth`` path.
        """
        torch.save(data.to_dict(), path)

    def _load_item(self, path: str) -> GemNetData:
        """Load one processed GemNet graph sample.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Reconstructed GemNet data object.
        """
        tensor_dict = torch.load(path, weights_only=True)
        return GemNetData(**tensor_dict)

    @property
    def signature(self) -> Config:
        """Dataset configuration signature.

        Returns:
            Configuration values that identify this GemNet dataset.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "cutoff": self.cutoff,
                "max_num_neighbors": self.max_num_neighbors,
                "graph_method": self.graph_method,
                "min_facet_area": self.min_facet_area,
                "cov_radii_scale": self.cov_radii_scale,
                "quadruplets": self.quadruplets,
                "int_cutoff": self.int_cutoff,
                "int_max_neighbors": self.int_max_neighbors,
                "int_graph_method": self.int_graph_method,
                "int_min_facet_area": self.int_min_facet_area,
                "int_cov_radii_scale": self.int_cov_radii_scale,
                "oc_mode": self.oc_mode,
                "oc_cutoff_aeaint": self.oc_cutoff_aeaint,
                "oc_cutoff_aint": self.oc_cutoff_aint,
                "oc_max_neighbors_aeaint": self.oc_max_neighbors_aeaint,
                "oc_max_neighbors_aint": self.oc_max_neighbors_aint,
                "oc_graph_method_aeaint": self.oc_graph_method_aeaint,
                "oc_min_facet_area_aeaint": self.oc_min_facet_area_aeaint,
                "oc_cov_radii_scale_aeaint": self.oc_cov_radii_scale_aeaint,
                "oc_graph_method_aint": self.oc_graph_method_aint,
                "oc_min_facet_area_aint": self.oc_min_facet_area_aint,
                "oc_cov_radii_scale_aint": self.oc_cov_radii_scale_aint,
            }
        )
        return signature
