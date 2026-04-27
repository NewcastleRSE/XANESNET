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

import numpy as np
import torch
from torch_geometric.data import Data

from xanesnet.datasets._mp import mp_save_path, run_mp_prepare
from xanesnet.datasets.base import Dataset as _BaseDataset
from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config
from xanesnet.utils.graph import build_absorber_paths, build_edges

from ...registry import DatasetRegistry
from ..e3ee_full import SPECTRUM_KEYS, E3EEFullDataset


@DatasetRegistry.register("e3ee_full_mp")
class E3EEFullDatasetMp(E3EEFullDataset):
    """Multiprocessing variant of :class:`E3EEFullDataset`."""

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
        num_workers: int | None,
    ) -> None:
        super().__init__(
            dataset_type=dataset_type,
            datasource=datasource,
            root=root,
            preload=preload,
            skip_prepare=skip_prepare,
            split_ratios=split_ratios,
            split_indexfile=split_indexfile,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            use_path_branch=use_path_branch,
            max_paths_per_site=max_paths_per_site,
            graph_method=graph_method,
            min_facet_area=min_facet_area,
            cov_radii_scale=cov_radii_scale,
        )
        self.num_workers = num_workers

    def prepare(self) -> bool:
        skip_processing = _BaseDataset.prepare(self)
        if skip_processing:
            return True

        self._length = run_mp_prepare(self, len(self.datasource), self.num_workers)
        return True

    def _process_range(self, start: int, end: int) -> None:
        for idx in range(start, end):
            pmg_obj = self.datasource[idx]
            for key in SPECTRUM_KEYS:
                if key in pmg_obj.site_properties.keys():
                    break
            else:
                logging.warning(
                    f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['file_name']}); skipping."
                )
                continue

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

            data_kwargs: dict = {
                "x": atomic_numbers,
                "absorber_mask": absorber_mask,
                "edge_src": edge_index[0],
                "edge_dst": edge_index[1],
                "edge_weight": edge_weight,
                "edge_vec": edge_vec,
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
                for site_idx in range(n_atoms_total):
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
            self._save_data(struct, mp_save_path(self.processed_dir, idx, 0))

    @property
    def signature(self) -> Config:
        return super().signature
