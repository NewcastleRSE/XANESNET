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
from typing import Protocol

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


# for typing
class E3EEFullBatch(Protocol):
    # Padded per-sample node fields [B, N_max, ...]
    x: torch.Tensor
    mask: torch.Tensor
    # [B, N_max] bool; True at atoms that carry a ground-truth spectrum
    absorber_mask: torch.Tensor
    # Flat edge fields, already offset into the padded B*N_max layout
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_weight: torch.Tensor
    edge_vec: torch.Tensor
    att_src: torch.Tensor
    att_dst: torch.Tensor
    att_dist: torch.Tensor
    att_vec: torch.Tensor
    # Flat per-site triplet scalars. ``path_center`` is the flat atom index of
    # the site a path belongs to (into the padded B*N_max layout).
    path_center: torch.Tensor
    path_j: torch.Tensor
    path_k: torch.Tensor
    path_r0j: torch.Tensor
    path_r0k: torch.Tensor
    path_rjk: torch.Tensor
    path_cosangle: torch.Tensor
    # Targets, concatenated over absorbers across the batch
    energies: torch.Tensor
    intensities: torch.Tensor
    file_name: list[str]


###############################################################################
#################################### CLASS ####################################
###############################################################################


@DatasetRegistry.register("e3ee_full")
class E3EEFullDataset(TorchGeometricDataset):
    """
    Full-structure E3EE dataset.

    Emits one sample per structure (periodic or non-periodic) and predicts a
    spectrum for every atom. Atoms that carry a ground-truth spectrum are
    flagged in ``absorber_mask``; the training loop selects those rows via the
    mask (same pattern as SchNet / DimeNet).

    All edges are computed once per structure. Absorber-centred triplet paths
    are computed for every site independently (with ``max_paths_per_site``
    paths each) and tagged with ``path_center`` so that the model can scatter
    them into the per-atom layout.
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
        max_paths_per_site: int,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
        att_cutoff: float,
        att_max_num_neighbors: int,
        att_graph_method: str,
        att_min_facet_area: float | str | None,
        att_cov_radii_scale: float,
        use_absorber_mask: bool,
    ) -> None:
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.use_path_branch = use_path_branch
        self.max_paths_per_site = max_paths_per_site
        self.graph_method = graph_method
        self.min_facet_area = min_facet_area
        self.cov_radii_scale = cov_radii_scale
        self.att_cutoff = att_cutoff
        self.att_max_num_neighbors = att_max_num_neighbors
        self.att_graph_method = att_graph_method
        self.att_min_facet_area = att_min_facet_area
        self.att_cov_radii_scale = att_cov_radii_scale
        self.use_absorber_mask = use_absorber_mask

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        pmg_obj = self.datasource[idx]
        for key in SPECTRUM_KEYS:
            if key in pmg_obj.site_properties.keys():
                break
        else:
            logging.warning(f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['file_name']}); skipping.")
            return 0

        xanes = np.array(pmg_obj.site_properties[key], dtype=object)
        absorber_idxs: list[int] = np.where(xanes != None)[0].tolist()  # noqa: E711

        n_atoms_total = len(pmg_obj)
        atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)

        absorber_mask = torch.zeros(n_atoms_total, dtype=torch.bool)
        for si in absorber_idxs:
            absorber_mask[si] = True

        energies_stack = torch.tensor(
            np.array([xanes[si]["energies"] for si in absorber_idxs], dtype=np.float32),
            dtype=torch.float32,
        )
        intensities_stack = torch.tensor(
            np.array([xanes[si]["intensities"] for si in absorber_idxs], dtype=np.float32),
            dtype=torch.float32,
        )

        edge_index, edge_weight, edge_vec, _edge_attr = build_edges(
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
        if self.use_absorber_mask:
            abs_self_idx = torch.tensor(absorber_idxs, dtype=torch.int64)
            src_full = att_edge_index[0].to(dtype=torch.int64)
            keep = absorber_mask[src_full]
            att_src = torch.cat([abs_self_idx, src_full[keep]], dim=0)
            att_dst = torch.cat(
                [abs_self_idx, att_edge_index[1].to(dtype=torch.int64)[keep]],
                dim=0,
            )
            att_dist = torch.cat(
                [
                    torch.zeros(abs_self_idx.shape[0], dtype=torch.float32),
                    att_edge_weight.to(dtype=torch.float32)[keep],
                ],
                dim=0,
            )
            att_vec = torch.cat(
                [
                    torch.zeros(abs_self_idx.shape[0], 3, dtype=torch.float32),
                    att_edge_vec.to(dtype=torch.float32)[keep],
                ],
                dim=0,
            )
        else:
            self_idx = torch.arange(n_atoms_total, dtype=torch.int64)
            att_src = torch.cat([self_idx, att_edge_index[0].to(dtype=torch.int64)], dim=0)
            att_dst = torch.cat([self_idx, att_edge_index[1].to(dtype=torch.int64)], dim=0)
            att_dist = torch.cat(
                [torch.zeros(n_atoms_total, dtype=torch.float32), att_edge_weight.to(dtype=torch.float32)],
                dim=0,
            )
            att_vec = torch.cat(
                [torch.zeros(n_atoms_total, 3, dtype=torch.float32), att_edge_vec.to(dtype=torch.float32)],
                dim=0,
            )

        data_kwargs: dict = {
            "x": atomic_numbers,
            "absorber_mask": absorber_mask,
            "edge_src": edge_index[0],
            "edge_dst": edge_index[1],
            "edge_weight": edge_weight,
            "edge_vec": edge_vec,
            "att_src": att_src,
            "att_dst": att_dst,
            "att_dist": att_dist,
            "att_vec": att_vec,
            "energies": energies_stack,
            "intensities": intensities_stack,
            "file_name": pmg_obj.properties["file_name"],
        }

        if self.use_path_branch:
            centers: list[torch.Tensor] = []
            j_list: list[torch.Tensor] = []
            k_list: list[torch.Tensor] = []
            r0j_list: list[torch.Tensor] = []
            r0k_list: list[torch.Tensor] = []
            rjk_list: list[torch.Tensor] = []
            cos_list: list[torch.Tensor] = []

            site_iter = absorber_idxs if self.use_absorber_mask else range(n_atoms_total)
            for site_idx in site_iter:
                paths = build_absorber_paths(
                    pmg_obj,
                    absorber_idx=site_idx,
                    cutoff=self.cutoff,
                    max_paths=self.max_paths_per_site,
                )
                n_p = paths["path_j"].shape[0]
                if n_p == 0:
                    continue
                centers.append(torch.full((n_p,), site_idx, dtype=torch.int64))
                j_list.append(paths["path_j"])
                k_list.append(paths["path_k"])
                r0j_list.append(paths["path_r0j"])
                r0k_list.append(paths["path_r0k"])
                rjk_list.append(paths["path_rjk"])
                cos_list.append(paths["path_cosangle"])

            if centers:
                data_kwargs["path_center"] = torch.cat(centers, dim=0)
                data_kwargs["path_j"] = torch.cat(j_list, dim=0)
                data_kwargs["path_k"] = torch.cat(k_list, dim=0)
                data_kwargs["path_r0j"] = torch.cat(r0j_list, dim=0)
                data_kwargs["path_r0k"] = torch.cat(r0k_list, dim=0)
                data_kwargs["path_rjk"] = torch.cat(rjk_list, dim=0)
                data_kwargs["path_cosangle"] = torch.cat(cos_list, dim=0)
            else:
                data_kwargs["path_center"] = torch.zeros(0, dtype=torch.int64)
                data_kwargs["path_j"] = torch.zeros(0, dtype=torch.int64)
                data_kwargs["path_k"] = torch.zeros(0, dtype=torch.int64)
                data_kwargs["path_r0j"] = torch.zeros(0, dtype=torch.float32)
                data_kwargs["path_r0k"] = torch.zeros(0, dtype=torch.float32)
                data_kwargs["path_rjk"] = torch.zeros(0, dtype=torch.float32)
                data_kwargs["path_cosangle"] = torch.zeros(0, dtype=torch.float32)

        struct = Data(**data_kwargs)
        self._save_data(struct, save_path_fn(0))
        return 1

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        """
        Pad node tensors to ``[B, N_max, ...]`` and offset flat edge/path
        atom indices (including ``path_center``) by ``b * N_max`` so they
        index into the padded layout used by the model.
        """
        bsz = len(batch)

        x_list = [sample.x for sample in batch]
        n_atoms_per_sample = torch.tensor([xi.shape[0] for xi in x_list], dtype=torch.int64)
        n_max = int(n_atoms_per_sample.max().item()) if bsz > 0 else 0

        x = pad_sequence(x_list, batch_first=True, padding_value=0)
        mask_list = [torch.ones(xi.shape[0], dtype=torch.bool) for xi in x_list]
        mask = pad_sequence(mask_list, batch_first=True, padding_value=False).to(dtype=torch.bool)

        absorber_mask_list = [s.absorber_mask.to(dtype=torch.bool) for s in batch]
        absorber_mask = pad_sequence(absorber_mask_list, batch_first=True, padding_value=False).to(dtype=torch.bool)

        # Concatenate per-absorber targets across the batch (align with
        # absorber_mask.view(-1) order: sample-major, atom-minor).
        intensities = torch.cat([s.intensities.to(dtype=torch.float32) for s in batch], dim=0)
        energies = torch.cat([s.energies.to(dtype=torch.float32) for s in batch], dim=0)

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

        att_src_list: list[torch.Tensor] = []
        att_dst_list: list[torch.Tensor] = []
        att_dist_list: list[torch.Tensor] = []
        att_vec_list: list[torch.Tensor] = []
        for b, sample in enumerate(batch):
            offset = b * n_max
            att_src_list.append(sample.att_src + offset)
            att_dst_list.append(sample.att_dst + offset)
            att_dist_list.append(sample.att_dist)
            att_vec_list.append(sample.att_vec)
        att_src = torch.cat(att_src_list, dim=0) if att_src_list else torch.zeros(0, dtype=torch.int64)
        att_dst = torch.cat(att_dst_list, dim=0) if att_dst_list else torch.zeros(0, dtype=torch.int64)
        att_dist = torch.cat(att_dist_list, dim=0) if att_dist_list else torch.zeros(0, dtype=torch.float32)
        att_vec = torch.cat(att_vec_list, dim=0) if att_vec_list else torch.zeros(0, 3, dtype=torch.float32)

        has_paths = all(hasattr(s, "path_j") for s in batch)
        path_center = torch.zeros(0, dtype=torch.int64)
        path_j = torch.zeros(0, dtype=torch.int64)
        path_k = torch.zeros(0, dtype=torch.int64)
        path_r0j = torch.zeros(0, dtype=torch.float32)
        path_r0k = torch.zeros(0, dtype=torch.float32)
        path_rjk = torch.zeros(0, dtype=torch.float32)
        path_cosangle = torch.zeros(0, dtype=torch.float32)
        if has_paths:
            pc_list: list[torch.Tensor] = []
            pj_list: list[torch.Tensor] = []
            pk_list: list[torch.Tensor] = []
            for b, sample in enumerate(batch):
                offset = b * n_max
                pc_list.append(sample.path_center + offset)
                pj_list.append(sample.path_j + offset)
                pk_list.append(sample.path_k + offset)
            path_center = torch.cat(pc_list, dim=0)
            path_j = torch.cat(pj_list, dim=0)
            path_k = torch.cat(pk_list, dim=0)
            path_r0j = torch.cat([s.path_r0j for s in batch], dim=0)
            path_r0k = torch.cat([s.path_r0k for s in batch], dim=0)
            path_rjk = torch.cat([s.path_rjk for s in batch], dim=0)
            path_cosangle = torch.cat([s.path_cosangle for s in batch], dim=0)

        file_name: list[str] = []
        for s in batch:
            file_name.extend([s.file_name] * int(s.absorber_mask.sum().item()))

        batched = Batch.from_data_list(
            batch,
            exclude_keys=[
                "x",
                "energies",
                "intensities",
                "absorber_mask",
                "edge_src",
                "edge_dst",
                "edge_weight",
                "edge_vec",
                "att_src",
                "att_dst",
                "att_dist",
                "att_vec",
                "path_center",
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
        setattr(batched, "absorber_mask", absorber_mask)
        setattr(batched, "edge_src", edge_src)
        setattr(batched, "edge_dst", edge_dst)
        setattr(batched, "edge_weight", edge_weight)
        setattr(batched, "edge_vec", edge_vec)
        setattr(batched, "att_src", att_src)
        setattr(batched, "att_dst", att_dst)
        setattr(batched, "att_dist", att_dist)
        setattr(batched, "att_vec", att_vec)
        setattr(batched, "path_center", path_center)
        setattr(batched, "path_j", path_j)
        setattr(batched, "path_k", path_k)
        setattr(batched, "path_r0j", path_r0j)
        setattr(batched, "path_r0k", path_r0k)
        setattr(batched, "path_rjk", path_rjk)
        setattr(batched, "path_cosangle", path_cosangle)
        setattr(batched, "energies", energies)
        setattr(batched, "intensities", intensities)
        setattr(batched, "file_name", file_name)

        return batched

    @staticmethod
    def _save_data(data: Data, path: str) -> None:
        tensor_dict = data.to_dict()
        torch.save(tensor_dict, path)

    def _load_item(self, path: str) -> Data:
        tensor_dict = torch.load(path, weights_only=True)
        return Data(**tensor_dict)

    @property
    def signature(self) -> Config:
        signature = super().signature
        signature.update_with_dict(
            {
                "cutoff": self.cutoff,
                "max_num_neighbors": self.max_num_neighbors,
                "use_path_branch": self.use_path_branch,
                "max_paths_per_site": self.max_paths_per_site,
                "graph_method": self.graph_method,
                "min_facet_area": self.min_facet_area,
                "cov_radii_scale": self.cov_radii_scale,
                "att_cutoff": self.att_cutoff,
                "att_max_num_neighbors": self.att_max_num_neighbors,
                "att_graph_method": self.att_graph_method,
                "att_min_facet_area": self.att_min_facet_area,
                "att_cov_radii_scale": self.att_cov_radii_scale,
                "use_absorber_mask": self.use_absorber_mask,
            }
        )
        return signature
