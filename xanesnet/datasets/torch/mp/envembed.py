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
from xanesnet.utils.math import gaussian_fit

from ...registry import DatasetRegistry
from ..envembed import SPECTRUM_KEYS, EnvEmbedData, EnvEmbedDataset


@DatasetRegistry.register("envembed_mp")
class EnvEmbedDatasetMp(EnvEmbedDataset):
    """Multiprocessing variant of :class:`EnvEmbedDataset`."""

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
        widths_eV: list[float],
        basis_stride: int,
        basis_path: str | None,
        env_radius: float | None,
        descriptors: list[Config],
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
            widths_eV=widths_eV,
            basis_stride=basis_stride,
            basis_path=basis_path,
            env_radius=env_radius,
            descriptors=descriptors,
        )
        self.num_workers = num_workers

    def prepare(self) -> bool:
        skip_processing = _BaseDataset.prepare(self)
        if skip_processing:
            return True

        assert self.basis is not None, "Spectral basis must be set up successfully."

        self._length = run_mp_prepare(self, len(self.datasource), self.num_workers)
        return True

    def _process_range(self, start: int, end: int) -> None:
        assert self.basis is not None
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

            descriptor_features = []
            for descriptor in self.descriptor_list:
                feature = descriptor.transform_pmg(pmg_obj, site_index=None)
                descriptor_features.append(feature)
            descriptor_features = np.concatenate(descriptor_features, axis=1)
            descriptor_features = torch.tensor(descriptor_features, dtype=torch.float32)

            seq = 0
            for site_idx in xanes_idxs:
                spectrum = pmg_obj.site_properties[key][site_idx]
                energies = torch.tensor(spectrum["energies"], dtype=torch.float32)
                intensities = torch.tensor(spectrum["intensities"], dtype=torch.float32)

                site_descs, site_dists = self._build_site_environment(
                    pmg_obj, absorber_idx=site_idx, all_descriptors=descriptor_features
                )

                c_star = gaussian_fit(basis=self.basis, xanes=intensities)

                data = EnvEmbedData(
                    descriptor_features=site_descs,
                    distance_features=site_dists,
                    intensities=intensities,
                    energies=energies,
                    c_star=c_star,
                    file_name=pmg_obj.properties["file_name"],
                    basis=self.basis,
                )

                data.save(mp_save_path(self.processed_dir, idx, seq))
                seq += 1

    @property
    def signature(self) -> Config:
        return super().signature
