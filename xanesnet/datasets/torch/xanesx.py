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
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.descriptors import Descriptor, DescriptorRegistry
from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.math import SpectralBasis, fft, gaussian_fit

from ..base import TorchDataset
from ..registry import DatasetRegistry

SPECTRUM_KEYS = ["XANES", "XANES_K"]  # TODO maybe put this somewhere more central?


@dataclass
class XanesXData:
    x: torch.Tensor | None = None
    y: torch.Tensor | None = None
    energies: torch.Tensor | None = None
    fourier: torch.Tensor | None = None
    c_star: torch.Tensor | None = None
    file_name: str | list[str] | None = None

    def to(self, device: str | torch.device) -> "XanesXData":
        # send batch do device
        for attr in ["x", "y", "fourier", "c_star", "energies"]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "energies": self.energies,
            "fourier": self.fourier,
            "c_star": self.c_star,
            "file_name": self.file_name,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "XanesXData":
        return cls(
            x=state.get("x"),
            y=state.get("y"),
            energies=state.get("energies"),
            fourier=state.get("fourier"),
            c_star=state.get("c_star"),
            file_name=state.get("file_name"),
        )

    def save(self, path: str) -> str:
        torch.save(self.to_state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str) -> "XanesXData":
        state = torch.load(path, weights_only=True)
        return cls.from_state_dict(state)


###############################################################################
#################################### CLASS ####################################
###############################################################################


@DatasetRegistry.register("xanesx")
class XanesXDataset(TorchDataset):
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
        # descriptors
        descriptors: list[Config],
    ) -> None:
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.mode = mode
        self.fourier = fourier
        self.fourier_concat = fourier_concat
        self.gaussian = gaussian
        self.widths_eV = widths_eV
        self.basis_stride = basis_stride
        self.basis_path = basis_path

        # Some assertions
        if self.fourier or self.gaussian:
            if self.mode != "forward":
                raise NotImplementedError("Fourier and Gaussian features are only allowed in FORWARD mode.")
            if self.fourier and self.gaussian:
                raise NotImplementedError("Fourier and Gaussian features cannot be used together.")

        # Create descriptors
        self.descriptor_configs = descriptors
        self.descriptor_list: list[Descriptor] = []
        descriptor_types = ", ".join(d.get_str("descriptor_type") for d in descriptors)
        logging.info(f"Initialising descriptors: {descriptor_types}")
        for descriptor_config in descriptors:
            descriptor_type = descriptor_config.get_str("descriptor_type")
            descriptor = DescriptorRegistry.get(descriptor_type)(**descriptor_config.as_kwargs())
            self.descriptor_list.append(descriptor)

        # Setup spectral basis only if needed
        self.basis: SpectralBasis | None = None
        if self.gaussian:
            self._setup_spectral_basis()

    def prepare(self) -> bool:
        skip_processing = super().prepare()
        if skip_processing:
            return True

        counter = 0  # Counter for naming processed files
        for idx, pmg_obj in tqdm(enumerate(self.datasource), desc="Processing data", total=len(self.datasource)):
            # Check if XANES spectrum is available for the sample; if not, skip processing
            for key in SPECTRUM_KEYS:
                if key in pmg_obj.site_properties.keys():
                    break
            else:
                logging.warning(
                    f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['file_name']}); skipping."
                )
                continue

            xanes = np.array(pmg_obj.site_properties[key], dtype=object)
            xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()

            # Compute descriptor features
            descriptor_features = []
            for descriptor in self.descriptor_list:
                feature = descriptor.transform_pmg(pmg_obj, site_index=xanes_idxs)
                descriptor_features.append(feature)
            descriptor_features = np.concatenate(descriptor_features, axis=1)

            for site_idx, df in zip(xanes_idxs, descriptor_features):
                # descriptor features
                df = torch.tensor(df, dtype=torch.float32)

                # XANES
                spectrum = pmg_obj.site_properties[key][site_idx]
                energies = torch.tensor(spectrum["energies"], dtype=torch.float32)
                intensities = torch.tensor(spectrum["intensities"], dtype=torch.float32)

                # FFT
                fourier = None
                if self.fourier:
                    fourier = fft(intensities, self.fourier_concat)

                # Gaussian
                c_star = None
                if self.gaussian:
                    assert self.basis is not None, "Spectral basis must be set up successfully before preparing data."
                    c_star = gaussian_fit(basis=self.basis, xanes=intensities)

                # Mode
                if self.mode == "forward":
                    x = df
                    y = intensities
                elif self.mode == "reverse":
                    x = intensities
                    y = df
                else:
                    raise ConfigError(f"Invalid mode: {self.mode}")

                # Create Data object
                data = XanesXData(
                    x=x,
                    y=y,
                    energies=energies,
                    fourier=fourier,
                    c_star=c_star,
                    file_name=pmg_obj.properties["file_name"],
                )

                # Save processed data
                save_path = os.path.join(self.processed_dir, f"{counter}.pth")
                data.save(save_path)
                counter += 1

        self._length = counter
        return True

    def _setup_spectral_basis(self) -> None:
        # Load directly from file
        if self.basis_path is not None:
            # TODO never tested this.
            self.basis = torch.load(self.basis_path)  # TODO: still uses torch.load without weights_only=True
            logging.info(f"Loaded spectral basis from file @ {self.basis_path}")
        # Create from datasource
        else:
            logging.info("Creating spectral basis from datasource")
            first_data = next(iter(self.datasource))
            # TODO requires same energy grid for all samples!
            for key in SPECTRUM_KEYS:
                if key in first_data.site_properties.keys():
                    break
            else:
                raise ValueError("No XANES spectrum found in datasource to set up spectral basis.")

            xanes = np.array(first_data.site_properties[key], dtype=object)
            xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()
            energies = torch.tensor(first_data.site_properties[key][xanes_idxs[0]]["energies"], dtype=torch.float32)
            self.basis = SpectralBasis(
                energies=energies,
                widths_eV=self.widths_eV,
                normalize_atoms=True,
                stride=self.basis_stride,
            )

    def collate_fn(self, batch: list[XanesXData]) -> XanesXData:
        """
        Collates a list of Data objects into a single Data object with batched tensors.
        """

        def _stack(tensors):
            if any(t is None for t in tensors):
                return None
            return torch.stack(tensors)

        return XanesXData(
            x=_stack([b.x for b in batch]),
            y=_stack([b.y for b in batch]),
            energies=_stack([b.energies for b in batch]),
            fourier=_stack([b.fourier for b in batch]),
            c_star=_stack([b.c_star for b in batch]),
            file_name=[b.file_name for b in batch],  # type: ignore[list-item]
        )

    def _load_item(self, path: str) -> XanesXData:
        return XanesXData.load(path)

    @property
    def signature(self) -> Config:
        """
        Return dataset signature as a dictionary.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "descriptors": self.descriptor_configs,
                "mode": self.mode,
                "fourier": self.fourier,
                "fourier_concat": self.fourier_concat,
                "gaussian": self.gaussian,
                "widths_eV": self.widths_eV,
                "basis_stride": self.basis_stride,
                "basis_path": self.basis_path,
            }
        )
        return signature
