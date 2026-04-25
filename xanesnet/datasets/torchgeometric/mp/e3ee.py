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
from ..e3ee import SPECTRUM_KEYS, E3EEDataset


@DatasetRegistry.register("e3ee_mp")
class E3EEDatasetMp(E3EEDataset):
    """Multiprocessing variant of :class:`E3EEDataset`. See the parent for semantics."""

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
            max_paths_per_structure=max_paths_per_structure,
            graph_method=graph_method,
            min_facet_area=min_facet_area,
            cov_radii_scale=cov_radii_scale,
        )
        self.num_workers = num_workers

    def prepare(self) -> bool:
        # Skip the parent's sequential prepare; run only the base directory
        # setup, then dispatch the per-sample work to a worker pool.
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

            atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)

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

            seq = 0
            for site_idx in absorber_idxs:
                spectrum = pmg_obj.site_properties[key][site_idx]
                energies = torch.tensor(spectrum["energies"], dtype=torch.float32)
                intensities = torch.tensor(spectrum["intensities"], dtype=torch.float32)

                data_kwargs: dict = {
                    "x": atomic_numbers,
                    "absorber_index": torch.tensor(site_idx, dtype=torch.int64),
                    "edge_src": edge_index[0],
                    "edge_dst": edge_index[1],
                    "edge_weight": edge_weight,
                    "edge_vec": edge_vec,
                    "energies": energies,
                    "intensities": intensities,
                    "file_name": f"{pmg_obj.properties['file_name']}::site_{site_idx}",
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
                self._save_data(struct, mp_save_path(self.processed_dir, idx, seq))
                seq += 1

    @property
    def signature(self) -> Config:
        return super().signature
