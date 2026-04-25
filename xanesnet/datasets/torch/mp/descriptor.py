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
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.math import fft, gaussian_fit

from ...registry import DatasetRegistry
from ..descriptor import SPECTRUM_KEYS, DescriptorData, DescriptorDataset


@DatasetRegistry.register("descriptor_mp")
class DescriptorDatasetMp(DescriptorDataset):
    """Multiprocessing variant of :class:`DescriptorDataset`."""

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
        mode: str,
        fourier: bool,
        fourier_concat: bool,
        gaussian: bool,
        widths_eV: list[float],
        basis_stride: int,
        basis_path: str | None,
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
            mode=mode,
            fourier=fourier,
            fourier_concat=fourier_concat,
            gaussian=gaussian,
            widths_eV=widths_eV,
            basis_stride=basis_stride,
            basis_path=basis_path,
            descriptors=descriptors,
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

            descriptor_features = []
            for descriptor in self.descriptor_list:
                feature = descriptor.transform_pmg(pmg_obj, site_index=xanes_idxs)
                descriptor_features.append(feature)
            descriptor_features = np.concatenate(descriptor_features, axis=1)

            seq = 0
            for site_idx, df in zip(xanes_idxs, descriptor_features):
                df = torch.tensor(df, dtype=torch.float32)

                spectrum = pmg_obj.site_properties[key][site_idx]
                energies = torch.tensor(spectrum["energies"], dtype=torch.float32)
                intensities = torch.tensor(spectrum["intensities"], dtype=torch.float32)

                fourier = None
                if self.fourier:
                    fourier = fft(intensities, self.fourier_concat)

                c_star = None
                if self.gaussian:
                    assert self.basis is not None
                    c_star = gaussian_fit(basis=self.basis, xanes=intensities)

                if self.mode == "forward":
                    x = df
                    y = intensities
                elif self.mode == "reverse":
                    x = intensities
                    y = df
                else:
                    raise ConfigError(f"Invalid mode: {self.mode}")

                data = DescriptorData(
                    x=x,
                    y=y,
                    energies=energies,
                    fourier=fourier,
                    c_star=c_star,
                    file_name=pmg_obj.properties["file_name"],
                )

                data.save(mp_save_path(self.processed_dir, idx, seq))
                seq += 1

    @property
    def signature(self) -> Config:
        return super().signature
