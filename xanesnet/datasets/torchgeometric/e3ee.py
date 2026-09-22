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

"""E3EE PyTorch Geometric dataset implementation."""

import logging
from typing import Any, Protocol

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from xanesnet.datasets.base import SavePathFn, TorchGeometricDataset
from xanesnet.datasources import DataSource
from xanesnet.graphs import GraphBuilderRegistry
from xanesnet.graphs.utils.target_site_paths import build_target_site_paths
from xanesnet.serialization.config import Config

from ..registry import DatasetRegistry


class E3EEBatch(Protocol):
    """Protocol for batches emitted by ``E3EEDataset.collate_fn``.

    Node fields are padded to ``(batch, max_nodes, ...)``. Flat edge and path
    indices are offset into the padded ``batch * max_nodes`` layout.
    """

    # Padded per-sample node fields [B, N_max, ...]
    x: torch.Tensor
    mask: torch.Tensor
    # [B] target-site index into the padded layout (0..N_max-1)
    target_site_index: torch.Tensor
    # Flat edge fields, already offset into the padded B*N_max layout
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_weight: torch.Tensor
    edge_vec: torch.Tensor
    att_dst: torch.Tensor
    att_dist: torch.Tensor
    att_vec: torch.Tensor
    # Flat target-site-centered triplet scalars, indices into padded layout
    path_j: torch.Tensor
    path_k: torch.Tensor
    path_r0j: torch.Tensor
    path_r0k: torch.Tensor
    path_rjk: torch.Tensor
    path_cosangle: torch.Tensor
    path_batch: torch.Tensor
    # Targets
    energies: torch.Tensor
    intensities: torch.Tensor
    sample_id: list[str]


@DatasetRegistry.register("e3ee")
class E3EEDataset(TorchGeometricDataset):
    """E3EE dataset that emits one graph sample per target site.

    The dataset preserves target-site ordering and supports both
    periodic Structures and non-periodic Molecules. All edges and target-site-
    centered triplet path scalars are precomputed at prepare() time using the
    shared graph utilities, so the model does not need to rebuild geometry at
    forward time.

    Both the main and attention graphs are constructed by
    :class:`~xanesnet.graphs.base.GraphBuilder` instances resolved from the
    ``graph_builder`` and ``att_graph_builder`` nested configs.

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw datasource of pymatgen structures or molecules.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        graph_builder: Main graph builder configuration.
        att_graph_builder: Attention graph builder configuration.
        use_path_branch: Whether to precompute target-site-centered paths. The
            path cutoff is taken from ``graph_builder.cutoff``.
        max_paths_per_structure: Maximum target-site paths saved per structure.
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
        att_graph_builder: Config,
        use_path_branch: bool,
        max_paths_per_structure: int,
    ) -> None:
        """Initialize the E3EE dataset."""
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.graph_builder_config = graph_builder
        self.graph_builder = GraphBuilderRegistry.create(
            graph_builder.get_str("graph_builder_type"), **graph_builder.as_kwargs()
        )
        self.att_graph_builder_config = att_graph_builder
        self.att_graph_builder = GraphBuilderRegistry.create(
            att_graph_builder.get_str("graph_builder_type"), **att_graph_builder.as_kwargs()
        )
        self.use_path_branch = use_path_branch
        self.max_paths_per_structure = max_paths_per_structure

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into target-site-centered graph samples.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps per-target-site sequence numbers to output paths.

        Returns:
            Number of target-site graph samples written.
        """
        pmg_obj = self.datasource[idx]
        if "spectrum" not in pmg_obj.site_properties:
            logging.warning(f"No spectrum found for sample {idx} ({pmg_obj.properties['sample_id']}); skipping.")
            return 0

        spectra = np.array(pmg_obj.site_properties["spectrum"], dtype=object)
        target_site_indices: list[int] = np.where(spectra != None)[0].tolist()  # noqa: E711

        atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)

        edge_index, edge_weight, edge_vec, _ = self.graph_builder.build(pmg_obj, compute_vectors=True)
        assert edge_vec is not None

        att_edge_index, att_edge_weight, att_edge_vec, _ = self.att_graph_builder.build(pmg_obj, compute_vectors=True)
        assert att_edge_vec is not None
        att_src_all = att_edge_index[0]
        att_dst_all = att_edge_index[1]

        seq = 0
        for site_idx in target_site_indices:
            spectrum = pmg_obj.site_properties["spectrum"][site_idx]
            energies = torch.tensor(spectrum["energies"], dtype=torch.float32)
            intensities = torch.tensor(spectrum["intensities"], dtype=torch.float32)

            sel = att_src_all == site_idx
            att_dst_site = torch.cat(
                [
                    torch.tensor([site_idx], dtype=torch.int64),
                    att_dst_all[sel].to(dtype=torch.int64),
                ],
                dim=0,
            )
            att_dist_site = torch.cat(
                [
                    torch.zeros(1, dtype=torch.float32),
                    att_edge_weight[sel].to(dtype=torch.float32),
                ],
                dim=0,
            )
            att_vec_site = torch.cat(
                [
                    torch.zeros(1, 3, dtype=torch.float32),
                    att_edge_vec[sel].to(dtype=torch.float32),
                ],
                dim=0,
            )

            data_kwargs: dict[str, Any] = {
                "x": atomic_numbers,
                "target_site_index": torch.tensor(site_idx, dtype=torch.int64),
                "edge_src": edge_index[0],
                "edge_dst": edge_index[1],
                "edge_weight": edge_weight,
                "edge_vec": edge_vec,
                "att_dst": att_dst_site,
                "att_dist": att_dist_site,
                "att_vec": att_vec_site,
                "energies": energies,
                "intensities": intensities,
                "sample_id": pmg_obj.properties["sample_id"],
            }

            if self.use_path_branch:
                paths = build_target_site_paths(
                    pmg_obj,
                    target_site_idx=site_idx,
                    cutoff=self.graph_builder.cutoff,
                    max_paths=self.max_paths_per_structure,
                )
                data_kwargs.update(paths)

            struct = Data(**data_kwargs)
            self._save_data(struct, save_path_fn(seq))
            seq += 1

        return seq

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        """Collate target-site-centered graph samples into one padded batch.

        Node tensors are padded to ``(batch, max_nodes, ...)`` and flat
        edge/path indices are offset by ``batch_index * max_nodes``.

        Args:
            batch: PyG graph samples loaded by ``__getitem__``.

        Returns:
            PyG batch with E3EE-specific padded tensors attached.
        """
        x_list = [sample.x for sample in batch]
        n_atoms_per_sample = torch.tensor([xi.shape[0] for xi in x_list], dtype=torch.int64)
        n_max = int(n_atoms_per_sample.max().item())

        x = pad_sequence(x_list, batch_first=True, padding_value=0)
        mask_list = [torch.ones(xi.shape[0], dtype=torch.bool) for xi in x_list]
        mask = pad_sequence(mask_list, batch_first=True, padding_value=False).to(dtype=torch.bool)

        intensities = torch.stack([s.intensities.to(dtype=torch.float32) for s in batch], dim=0)
        energies = torch.stack([s.energies.to(dtype=torch.float32) for s in batch], dim=0)

        target_site_index = torch.stack([s.target_site_index.to(dtype=torch.int64).reshape(()) for s in batch], dim=0)

        edge_src_list: list[torch.Tensor] = []
        edge_dst_list: list[torch.Tensor] = []
        edge_weight_list: list[torch.Tensor] = []
        edge_vec_list: list[torch.Tensor] = []
        for b, sample in enumerate(batch):
            offset = b * n_max
            edge_src_list.append(sample.edge_src + offset)
            edge_dst_list.append(sample.edge_dst + offset)
            edge_weight_list.append(sample.edge_weight)
            edge_vec_list.append(sample.edge_vec)

        edge_src = torch.cat(edge_src_list, dim=0)
        edge_dst = torch.cat(edge_dst_list, dim=0)
        edge_weight = torch.cat(edge_weight_list, dim=0)
        edge_vec = torch.cat(edge_vec_list, dim=0)

        has_paths = all(hasattr(s, "path_j") for s in batch)
        path_j = torch.zeros(0, dtype=torch.int64)
        path_k = torch.zeros(0, dtype=torch.int64)
        path_r0j = torch.zeros(0, dtype=torch.float32)
        path_r0k = torch.zeros(0, dtype=torch.float32)
        path_rjk = torch.zeros(0, dtype=torch.float32)
        path_cosangle = torch.zeros(0, dtype=torch.float32)
        path_batch = torch.zeros(0, dtype=torch.int64)
        if has_paths:
            path_j_list: list[torch.Tensor] = []
            path_k_list: list[torch.Tensor] = []
            path_batch_list: list[torch.Tensor] = []
            for b, sample in enumerate(batch):
                offset = b * n_max
                path_j_list.append(sample.path_j + offset)
                path_k_list.append(sample.path_k + offset)
                path_batch_list.append(torch.full((sample.path_j.shape[0],), b, dtype=torch.int64))
            path_j = torch.cat(path_j_list, dim=0)
            path_k = torch.cat(path_k_list, dim=0)
            path_batch = torch.cat(path_batch_list, dim=0)
            path_r0j = torch.cat([s.path_r0j for s in batch], dim=0)
            path_r0k = torch.cat([s.path_r0k for s in batch], dim=0)
            path_rjk = torch.cat([s.path_rjk for s in batch], dim=0)
            path_cosangle = torch.cat([s.path_cosangle for s in batch], dim=0)

        sample_id: list[str] = [s.sample_id for s in batch]

        batched = Batch.from_data_list(
            batch,
            exclude_keys=[
                "x",
                "energies",
                "intensities",
                "target_site_index",
                "edge_src",
                "edge_dst",
                "edge_weight",
                "edge_vec",
                "path_j",
                "path_k",
                "path_r0j",
                "path_r0k",
                "path_rjk",
                "path_cosangle",
                "sample_id",
            ],
        )

        setattr(batched, "x", x)
        setattr(batched, "mask", mask)
        setattr(batched, "target_site_index", target_site_index)
        setattr(batched, "edge_src", edge_src)
        setattr(batched, "edge_dst", edge_dst)
        setattr(batched, "edge_weight", edge_weight)
        setattr(batched, "edge_vec", edge_vec)
        setattr(batched, "path_j", path_j)
        setattr(batched, "path_k", path_k)
        setattr(batched, "path_r0j", path_r0j)
        setattr(batched, "path_r0k", path_r0k)
        setattr(batched, "path_rjk", path_rjk)
        setattr(batched, "path_cosangle", path_cosangle)
        setattr(batched, "path_batch", path_batch)
        setattr(batched, "energies", energies)
        setattr(batched, "intensities", intensities)
        setattr(batched, "sample_id", sample_id)

        return batched

    @staticmethod
    def _save_data(data: Data, path: str) -> None:
        """Save one PyG data object as a tensor dictionary.

        Args:
            data: Data object to serialize.
            path: Destination ``.pth`` path.
        """
        tensor_dict = data.to_dict()
        torch.save(tensor_dict, path)

    def _load_item(self, path: str) -> Data:
        """Load one processed E3EE graph sample.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Reconstructed PyG data object.
        """
        tensor_dict = torch.load(path, weights_only=True)
        return Data(**tensor_dict)

    @property
    def signature(self) -> Config:
        """Dataset configuration signature.

        Returns:
            Configuration values that identify this E3EE dataset.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "graph_builder": self.graph_builder_config,
                "att_graph_builder": self.att_graph_builder_config,
                "use_path_branch": self.use_path_branch,
                "max_paths_per_structure": self.max_paths_per_structure,
            }
        )
        return signature
