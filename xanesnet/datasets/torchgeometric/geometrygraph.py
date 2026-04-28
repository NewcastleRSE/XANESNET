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

"""Geometry graph PyTorch Geometric dataset implementation."""

import logging
from typing import Any, Protocol

import numpy as np
import torch
from pymatgen.core import Molecule, Structure
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config
from xanesnet.utils.graph import build_edges, compute_triplets_and_angles

from ..base import SavePathFn, TorchGeometricDataset
from ..registry import DatasetRegistry

SPECTRUM_KEYS = ["XANES", "XANES_K"]  # TODO maybe put this somewhere more central?


class GeometryGraphData(Data):
    """PyG data object with custom batching increments for triplet indices."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Return PyG batching increments for graph index fields.

        Args:
            key: Data attribute currently being batched.
            value: Attribute value currently being batched.
            *args: Additional PyG arguments.
            **kwargs: Additional PyG keyword arguments.

        Returns:
            Increment used by PyG for ``key``.
        """
        if key in ("idx_kj", "idx_ji"):
            edge_index = self.edge_index
            return edge_index.size(1) if edge_index is not None else 0
        return super().__inc__(key, value, *args, **kwargs)


class GeometryGraphBatch(Protocol):
    """Protocol for batches emitted by ``GeometryGraphDataset.collate_fn``."""

    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    batch: torch.Tensor
    # Triplet fields (only present when compute_angles=True)
    angle: torch.Tensor
    idx_kj: torch.Tensor
    idx_ji: torch.Tensor
    # Targets
    energies: torch.Tensor
    intensities: torch.Tensor
    absorber_mask: torch.Tensor
    file_name: list[str]


@DatasetRegistry.register("geometrygraph")
class GeometryGraphDataset(TorchGeometricDataset):
    """Geometry-based graph dataset.

    Supports two edge construction methods:

    - ``graph_method="radius"``: distance-cutoff radius graph.
    - ``graph_method="voronoi"``: Voronoi-tessellation graph (still bounded
      by ``cutoff``; Voronoi neighbours with distances above ``cutoff`` are
      dropped).

    In both cases ``edge_weight`` is the Cartesian edge length, and the
    returned graph is bidirectional (see ``xanesnet.utils.graph``).
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
        # params
        cutoff: float,
        max_num_neighbors: int,
        compute_angles: bool,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
    ) -> None:
        """Initialize the geometry graph dataset.

        Args:
            dataset_type: Registered dataset type name.
            datasource: Raw datasource of pymatgen structures or molecules.
            root: Directory that stores processed ``.pth`` files.
            preload: Whether to preload processed samples.
            skip_prepare: Whether to reuse existing processed files.
            split_ratios: Optional split ratios.
            split_indexfile: Optional path to split indices.
            cutoff: Graph cutoff in **Angstrom**.
            max_num_neighbors: Per-source neighbor cap.
            compute_angles: Whether to precompute triplet angles.
            graph_method: Graph construction method.
            min_facet_area: Optional Voronoi facet-area threshold.
            cov_radii_scale: Covalent-radii scale for graph construction.
        """
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.compute_angles = compute_angles
        self.graph_method = graph_method
        self.min_facet_area = min_facet_area
        self.cov_radii_scale = cov_radii_scale

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into one geometry graph sample.

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
            logging.warning(f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['file_name']}); skipping.")
            return 0

        xanes = np.array(pmg_obj.site_properties[key], dtype=object)
        xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()  # noqa: E711
        xanes = xanes[xanes_idxs]
        absorber_mask = torch.zeros(len(pmg_obj.labels), dtype=torch.bool)
        absorber_mask[xanes_idxs] = True
        intensities_np = np.array([x["intensities"] for x in xanes], dtype=np.float32)
        energies_np = np.array([x["energies"] for x in xanes], dtype=np.float32)

        atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
        cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)
        energies = torch.tensor(energies_np, dtype=torch.float32)
        intensities = torch.tensor(intensities_np, dtype=torch.float32)

        edge_index, edge_weight, angle, idx_kj, idx_ji = self._build_edges(
            pmg_obj,
            self.cutoff,
            self.max_num_neighbors,
            self.compute_angles,
            self.graph_method,
            self.min_facet_area,
            self.cov_radii_scale,
        )

        struct = GeometryGraphData(
            x=atomic_numbers,
            pos=cart_coords,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=None,
            angle=angle,
            idx_kj=idx_kj,
            idx_ji=idx_ji,
            energies=energies,
            intensities=intensities,
            absorber_mask=absorber_mask,
            file_name=pmg_obj.properties["file_name"],
        )

        self._save_data(struct, save_path_fn(0))
        return 1

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        """Collate geometry graph samples into one PyG batch.

        Args:
            batch: Geometry graph samples loaded by ``__getitem__``.

        Returns:
            PyG batch with target tensors concatenated over absorber sites.
        """
        fields_to_cat = ["energies", "intensities", "absorber_mask"]
        batched = Batch.from_data_list(batch, exclude_keys=fields_to_cat)
        for field in fields_to_cat:
            setattr(batched, field, torch.cat([getattr(d, field) for d in batch], dim=0))
        return batched

    @staticmethod
    def _build_edges(
        pmg_obj: Structure | Molecule,
        cutoff: float,
        max_num_neighbors: int,
        compute_angles: bool,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Build graph edges and optional triplet-angle tensors.

        Args:
            pmg_obj: Structure or molecule to convert to a graph.
            cutoff: Edge cutoff in **Angstrom**.
            max_num_neighbors: Per-source neighbor cap.
            compute_angles: Whether to compute triplet angle tensors.
            graph_method: Graph construction method.
            min_facet_area: Optional Voronoi facet-area threshold.
            cov_radii_scale: Covalent-radii scale for graph construction.

        Returns:
            ``(edge_index, edge_weight, angle, idx_kj, idx_ji)``. Angle and
            triplet index tensors are ``None`` when ``compute_angles`` is false.
        """
        edge_index, edge_weight, edge_vec, _edge_attr = build_edges(
            pmg_obj,
            cutoff,
            max_num_neighbors,
            compute_vectors=compute_angles,
            method=graph_method,
            min_facet_area=min_facet_area,
            cov_radii_scale=cov_radii_scale,
        )

        if not compute_angles:
            return edge_index, edge_weight, None, None, None

        assert edge_vec is not None
        is_periodic = isinstance(pmg_obj, Structure)
        angle, idx_kj, idx_ji = compute_triplets_and_angles(
            edge_index, edge_vec, num_nodes=len(pmg_obj), is_periodic=is_periodic
        )
        return edge_index, edge_weight, angle, idx_kj, idx_ji

    @staticmethod
    def _save_data(data: Data, path: str) -> None:
        """Save one PyG data object as a tensor dictionary.

        Args:
            data: Data object to serialize.
            path: Destination ``.pth`` path.
        """
        tensor_dict = data.to_dict()
        torch.save(tensor_dict, path)

    def _load_item(self, path: str) -> GeometryGraphData:
        """Load one processed geometry graph sample.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Reconstructed geometry graph data object.
        """
        tensor_dict = torch.load(path, weights_only=True)
        return GeometryGraphData(**tensor_dict)

    @property
    def signature(self) -> Config:
        """Dataset configuration signature.

        Returns:
            Configuration values that identify this geometry graph dataset.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "cutoff": self.cutoff,
                "max_num_neighbors": self.max_num_neighbors,
                "compute_angles": self.compute_angles,
                "graph_method": self.graph_method,
                "min_facet_area": self.min_facet_area,
                "cov_radii_scale": self.cov_radii_scale,
            }
        )
        return signature
