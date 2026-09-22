# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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
#
# Citations:
#   ...

"""Geometry graph PyTorch Geometric dataset implementation."""

import logging
from typing import Any

import numpy as np
import torch
from pymatgen.core import Molecule, Structure
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from xanesnet.datasources import DataSource
from xanesnet.graphs import GraphBuilder, GraphBuilderRegistry
from xanesnet.graphs.utils.triplets import compute_triplets_and_angles
from xanesnet.serialization.config import Config

from ..base import SavePathFn, TorchGeometricDataset
from ..registry import DatasetRegistry


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
            assert self.edge_index is not None
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)


class GeometryGraphBatch(Batch):
    """Typed PyG batch emitted by ``GeometryGraphDataset.collate_fn``.

    Calling :meth:`~torch_geometric.data.Batch.from_data_list` on this class
    preserves the concrete batch type while PyG adds its batching metadata.
    """

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
    target_site_mask: torch.Tensor
    target_site_index: torch.Tensor
    sample_id: list[str]


@DatasetRegistry.register("geometrygraph")
class GeometryGraphDataset(TorchGeometricDataset):
    """Geometry-based graph dataset.

    The graph is constructed by a :class:`~xanesnet.graphs.base.GraphBuilder`
    resolved from the ``graph_builder`` nested config. Every registered
    builder (``radius``, ``cov_radius``, ``voronoi``) guarantees a
    bidirectional edge list with Cartesian edge lengths in ``edge_weight``.

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw datasource of pymatgen structures or molecules.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        graph_builder: Graph builder configuration (contains
            ``graph_builder_type`` plus per-type parameters such as
            ``cutoff`` and ``max_num_neighbors``).
        compute_angles: Whether to precompute triplet angles.
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
        graph_builder: Config,
        compute_angles: bool,
    ) -> None:
        """Initialize the geometry graph dataset."""
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.graph_builder_config = graph_builder
        self.graph_builder = GraphBuilderRegistry.create(
            graph_builder.get_str("graph_builder_type"), **graph_builder.as_kwargs()
        )
        self.compute_angles = compute_angles

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into one geometry graph sample.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps sample sequence numbers to output paths.

        Returns:
            ``1`` when the graph was saved, otherwise ``0`` when skipped.
        """
        pmg_obj = self.datasource[idx]
        if "spectrum" not in pmg_obj.site_properties:
            logging.warning(f"No spectrum found for sample {idx} ({pmg_obj.properties['sample_id']}); skipping.")
            return 0

        spectra = np.array(pmg_obj.site_properties["spectrum"], dtype=object)
        target_site_indices: list[int] = np.where(spectra != None)[0].tolist()  # noqa: E711
        spectra = spectra[target_site_indices]
        target_site_mask = torch.zeros(len(pmg_obj.labels), dtype=torch.bool)
        target_site_mask[target_site_indices] = True
        intensities_np = np.array([x["intensities"] for x in spectra], dtype=np.float32)
        energies_np = np.array([x["energies"] for x in spectra], dtype=np.float32)

        atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
        cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)
        energies = torch.tensor(energies_np, dtype=torch.float32)
        intensities = torch.tensor(intensities_np, dtype=torch.float32)

        edge_index, edge_weight, angle, idx_kj, idx_ji = self._build_edges(
            pmg_obj,
            self.graph_builder,
            self.compute_angles,
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
            target_site_mask=target_site_mask,
            target_site_index=torch.tensor(target_site_indices, dtype=torch.int64),
            sample_id=pmg_obj.properties["sample_id"],
        )

        self._save_data(struct, save_path_fn(0))
        return 1

    def collate_fn(self, batch: list[BaseData]) -> GeometryGraphBatch:
        """Collate geometry graph samples into one PyG batch.

        Args:
            batch: Geometry graph samples loaded by ``__getitem__``.

        Returns:
            PyG batch with target tensors concatenated over target sites.
            ``sample_id`` is expanded per target site so every row in
            ``intensities`` / ``target_site_mask`` has a corresponding
            identifier.
        """
        fields_to_cat = ["energies", "intensities", "target_site_mask", "target_site_index"]
        batched = GeometryGraphBatch.from_data_list(batch, exclude_keys=[*fields_to_cat, "sample_id"])
        for field in fields_to_cat:
            values = [getattr(data, field) for data in batch]
            setattr(batched, field, torch.cat(values, dim=0))
        batched.sample_id = [
            str(getattr(data, "sample_id"))
            for data in batch
            for _ in range(int(getattr(data, "target_site_mask").sum().item()))
        ]
        return batched

    @staticmethod
    def _build_edges(
        pmg_obj: Structure | Molecule,
        graph_builder: GraphBuilder,
        compute_angles: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Build graph edges and optional triplet-angle tensors.

        Args:
            pmg_obj: Structure or molecule to convert to a graph.
            graph_builder: Fully initialized graph builder used to construct
                the edge list.
            compute_angles: Whether to compute triplet angle tensors.

        Returns:
            ``(edge_index, edge_weight, angle, idx_kj, idx_ji)``. Angle and
            triplet index tensors are ``None`` when ``compute_angles`` is false.
        """
        edge_index, edge_weight, edge_vec, _edge_attr = graph_builder.build(pmg_obj, compute_vectors=compute_angles)

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
                "graph_builder": self.graph_builder_config,
                "compute_angles": self.compute_angles,
            }
        )
        return signature
