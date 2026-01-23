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
from xanesnet.descriptors import DescriptorRegistry
from xanesnet.utils.math import SpectralBasis, fft, gaussian_fit

from .registry import DatasetRegistry
from .torch_dataset import TorchDataset


@dataclass
class Data:
    x: torch.Tensor | None = None
    y: torch.Tensor | None = None
    e: torch.Tensor | None = None
    fourier: torch.Tensor | None = None
    c_star: torch.Tensor | None = None
    file_name: str | list[Any] | None = None

    def to(self, device: str | torch.device) -> "Data":
        # send batch do device
        for attr in ["x", "y", "fourier", "c_star"]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self


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
        # params:
        mode: str,
        fourier: bool,
        fourier_concat: bool,
        gaussian: bool,
        widths_eV: list[float],
        basis_stride: int,
        basis_path: str | None,
        # descriptors
        descriptors: list[dict[str, Any]],
    ) -> None:
        super().__init__(dataset_type, datasource, root, preload)

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
        self.descriptor_list = []
        descriptor_types = ", ".join(d["descriptor_type"] for d in descriptors)
        logging.info(f"Initialising descriptors: {descriptor_types}")
        for descriptor_config in descriptors:
            descriptor = DescriptorRegistry.get(descriptor_config["descriptor_type"])(
                **descriptor_config.get("params", {})
            )
            self.descriptor_list.append(descriptor)

        # Setup spectral basis only if needed
        if self.gaussian:
            self._setup_spectral_basis()

    def process(self) -> bool:
        already_processed = super().process()
        if already_processed:
            return True

        for idx, data in tqdm(enumerate(self.datasource), desc="Processing data", total=len(self.datasource)):
            # Compute descriptor features
            descriptor_features = []
            for descriptor in self.descriptor_list:
                feature = descriptor.transform_pmg(data)
                descriptor_features.append(feature)
            descriptor_features = np.concatenate(descriptor_features, axis=0)
            descriptor_features = torch.tensor(descriptor_features, dtype=torch.float32)

            # XANES (first atom)
            energies, intensities = (
                data.site_properties["XANES"][0]["energies"],
                data.site_properties["XANES"][0]["intensities"],
            )
            energies = torch.tensor(energies, dtype=torch.float32)
            intensities = torch.tensor(intensities, dtype=torch.float32)

            # FFT
            fourier = None
            if self.fourier:
                fourier = fft(intensities, self.fourier_concat)

            # Gaussian
            c_star = None
            if self.gaussian:
                c_star = gaussian_fit(basis=self.basis, xanes=intensities)

            # Mode
            if self.mode == "forward":
                x = descriptor_features
                y = intensities
            elif self.mode == "reverse":
                x = intensities
                y = descriptor_features
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            # Create Data object
            data = Data(x=x, y=y, e=energies, fourier=fourier, c_star=c_star, file_name=data.properties["file_name"])

            # Save processed data
            save_path = os.path.join(self.processed_dir, f"{idx}.pt")
            torch.save(data, save_path)

        return True

    def _setup_spectral_basis(self) -> None:
        # Load directly from file
        if self.basis_path is not None:
            self.basis = torch.load(self.basis_path)
            logging.info(f"Loaded spectral basis from file @ {self.basis_path}")
        # Create from datasource
        else:
            logging.info("Creating spectral basis from datasource")
            first_data = next(iter(self.datasource))
            # ? Does this require that every XANES spectrum has the same energy grid?
            energies = first_data.site_properties["XANES"][0]["energies"]
            self.basis = SpectralBasis(
                energies=energies,
                widths_eV=self.widths_eV,
                normalize_atoms=True,
                stride=self.basis_stride,
            )

    def collate_fn(self, batch: list[Data]) -> Data:
        """
        Collates a list of Data objects into a single Data object with batched tensors.
        """

        def _stack(tensors):
            if any(t is None for t in tensors):
                return None
            return torch.stack(tensors)

        return Data(
            x=_stack([b.x for b in batch]),
            y=_stack([b.y for b in batch]),
            e=_stack([b.e for b in batch]),
            fourier=_stack([b.fourier for b in batch]),
            c_star=_stack([b.c_star for b in batch]),
            file_name=[b.file_name for b in batch],  # keep as list
        )

    @property
    def signature(self) -> dict[str, Any]:
        """
        Return dataset signature as a dictionary.
        """
        signature = super().signature
        signature.update(
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
