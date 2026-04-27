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

from xanesnet.datasets._mp import mp_save_path, run_mp_prepare
from xanesnet.datasets.base import Dataset as _BaseDataset
from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ...registry import DatasetRegistry
from ..geometrygraph import SPECTRUM_KEYS, GeometryGraphData, GeometryGraphDataset


@DatasetRegistry.register("geometrygraph_mp")
class GeometryGraphDatasetMp(GeometryGraphDataset):
    """Multiprocessing variant of :class:`GeometryGraphDataset`."""

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
            compute_angles=compute_angles,
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

            self._save_data(struct, mp_save_path(self.processed_dir, idx, 0))

    @property
    def signature(self) -> Config:
        return super().signature
