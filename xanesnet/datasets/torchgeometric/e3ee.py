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
from xanesnet.serialization.config import Config
from xanesnet.utils.graph import build_absorber_paths, build_edges

from ..registry import DatasetRegistry

SPECTRUM_KEYS = ["XANES", "XANES_K"]


class E3EEBatch(Protocol):
    """Protocol for batches emitted by ``E3EEDataset.collate_fn``.

    Node fields are padded to ``(batch, max_nodes, ...)``. Flat edge and path
    indices are offset into the padded ``batch * max_nodes`` layout.
    """

    # Padded per-sample node fields [B, N_max, ...]
    x: torch.Tensor
    mask: torch.Tensor
    # [B] absorber index into the padded layout (0..N_max-1)
    absorber_index: torch.Tensor
    # Flat edge fields, already offset into the padded B*N_max layout
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_weight: torch.Tensor
    edge_vec: torch.Tensor
    att_dst: torch.Tensor
    att_dist: torch.Tensor
    att_vec: torch.Tensor
    # Flat absorber-centred triplet scalars, indices into padded layout
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
    file_name: list[str]


@DatasetRegistry.register("e3ee")
class E3EEDataset(TorchGeometricDataset):
    """E3EE dataset that emits one graph sample per absorber site.

    The dataset preserves absorber ordering and supports both
    periodic Structures and non-periodic Molecules. All edges and absorber-
    centred triplet path scalars are precomputed at prepare() time using the
    shared graph utilities, so the model does not need to rebuild geometry at
    forward time.
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
        use_path_branch: bool,
        max_paths_per_structure: int,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
        att_cutoff: float,
        att_max_num_neighbors: int,
        att_graph_method: str,
        att_min_facet_area: float | str | None,
        att_cov_radii_scale: float,
    ) -> None:
        """Initialize the E3EE dataset.

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
            use_path_branch: Whether to precompute absorber-centered paths.
            max_paths_per_structure: Maximum absorber paths saved per structure.
            graph_method: Main graph construction method.
            min_facet_area: Optional Voronoi facet-area threshold.
            cov_radii_scale: Covalent-radii scale for graph construction.
            att_cutoff: Attention graph cutoff in **Angstrom**.
            att_max_num_neighbors: Attention graph per-source neighbor cap.
            att_graph_method: Attention graph construction method.
            att_min_facet_area: Optional attention Voronoi facet-area threshold.
            att_cov_radii_scale: Attention graph covalent-radii scale.
        """
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.use_path_branch = use_path_branch
        self.max_paths_per_structure = max_paths_per_structure
        self.graph_method = graph_method
        self.min_facet_area = min_facet_area
        self.cov_radii_scale = cov_radii_scale
        self.att_cutoff = att_cutoff
        self.att_max_num_neighbors = att_max_num_neighbors
        self.att_graph_method = att_graph_method
        self.att_min_facet_area = att_min_facet_area
        self.att_cov_radii_scale = att_cov_radii_scale

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into absorber-centered graph samples.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps per-absorber sequence numbers to output paths.

        Returns:
            Number of absorber graph samples written.
        """
        pmg_obj = self.datasource[idx]
        for key in SPECTRUM_KEYS:
            if key in pmg_obj.site_properties.keys():
                break
        else:
            logging.warning(f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['file_name']}); skipping.")
            return 0

        xanes = np.array(pmg_obj.site_properties[key], dtype=object)
        absorber_idxs: list[int] = np.where(xanes != None)[0].tolist()  # noqa: E711

        atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)

        edge_index, edge_weight, edge_vec, _ = build_edges(
            pmg_obj,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
            compute_vectors=True,
            method=self.graph_method,
            min_facet_area=self.min_facet_area,
            cov_radii_scale=self.cov_radii_scale,
        )
        assert edge_vec is not None

        att_edge_index, att_edge_weight, att_edge_vec, _ = build_edges(
            pmg_obj,
            cutoff=self.att_cutoff,
            max_num_neighbors=self.att_max_num_neighbors,
            compute_vectors=True,
            method=self.att_graph_method,
            min_facet_area=self.att_min_facet_area,
            cov_radii_scale=self.att_cov_radii_scale,
        )
        assert att_edge_vec is not None
        att_src_all = att_edge_index[0]
        att_dst_all = att_edge_index[1]

        seq = 0
        for site_idx in absorber_idxs:
            spectrum = pmg_obj.site_properties[key][site_idx]
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
                "absorber_index": torch.tensor(site_idx, dtype=torch.int64),
                "edge_src": edge_index[0],
                "edge_dst": edge_index[1],
                "edge_weight": edge_weight,
                "edge_vec": edge_vec,
                "att_dst": att_dst_site,
                "att_dist": att_dist_site,
                "att_vec": att_vec_site,
                "energies": energies,
                "intensities": intensities,
                "file_name": pmg_obj.properties["file_name"],
            }

            if self.use_path_branch:
                paths = build_absorber_paths(
                    pmg_obj,
                    absorber_idx=site_idx,
                    cutoff=self.cutoff,
                    max_paths=self.max_paths_per_structure,
                )
                data_kwargs.update(paths)

            struct = Data(**data_kwargs)
            self._save_data(struct, save_path_fn(seq))
            seq += 1

        return seq

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        """Collate absorber-centered graph samples into one padded batch.

        Node tensors are padded to ``(batch, max_nodes, ...)`` and flat
        edge/path indices are offset by ``batch_index * max_nodes``.

        Args:
            batch: PyG graph samples loaded by ``__getitem__``.

        Returns:
            PyG batch with E3EE-specific padded tensors attached.
        """
        bsz = len(batch)

        x_list = [sample.x for sample in batch]
        n_atoms_per_sample = torch.tensor([xi.shape[0] for xi in x_list], dtype=torch.int64)
        n_max = int(n_atoms_per_sample.max().item()) if bsz > 0 else 0

        x = pad_sequence(x_list, batch_first=True, padding_value=0)
        mask_list = [torch.ones(xi.shape[0], dtype=torch.bool) for xi in x_list]
        mask = pad_sequence(mask_list, batch_first=True, padding_value=False).to(dtype=torch.bool)

        intensities = torch.stack([s.intensities.to(dtype=torch.float32) for s in batch], dim=0)
        energies = torch.stack([s.energies.to(dtype=torch.float32) for s in batch], dim=0)

        absorber_index = torch.stack([s.absorber_index.to(dtype=torch.int64).reshape(()) for s in batch], dim=0)

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

        edge_src = torch.cat(edge_src_list, dim=0) if edge_src_list else torch.zeros(0, dtype=torch.int64)
        edge_dst = torch.cat(edge_dst_list, dim=0) if edge_dst_list else torch.zeros(0, dtype=torch.int64)
        edge_weight = torch.cat(edge_weight_list, dim=0) if edge_weight_list else torch.zeros(0, dtype=torch.float32)
        edge_vec = torch.cat(edge_vec_list, dim=0) if edge_vec_list else torch.zeros(0, 3, dtype=torch.float32)

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

        file_name: list[str] = [s.file_name for s in batch]

        batched = Batch.from_data_list(
            batch,
            exclude_keys=[
                "x",
                "energies",
                "intensities",
                "absorber_index",
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
                "file_name",
            ],
        )

        setattr(batched, "x", x)
        setattr(batched, "mask", mask)
        setattr(batched, "absorber_index", absorber_index)
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
        setattr(batched, "file_name", file_name)

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
                "cutoff": self.cutoff,
                "max_num_neighbors": self.max_num_neighbors,
                "use_path_branch": self.use_path_branch,
                "max_paths_per_structure": self.max_paths_per_structure,
                "graph_method": self.graph_method,
                "min_facet_area": self.min_facet_area,
                "cov_radii_scale": self.cov_radii_scale,
                "att_cutoff": self.att_cutoff,
                "att_max_num_neighbors": self.att_max_num_neighbors,
                "att_graph_method": self.att_graph_method,
                "att_min_facet_area": self.att_min_facet_area,
                "att_cov_radii_scale": self.att_cov_radii_scale,
            }
        )
        return signature
