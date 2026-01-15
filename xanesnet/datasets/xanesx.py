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

import numpy as np
import torch
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.descriptors import DescriptorRegistry
from xanesnet.utils.fourier import fft
from xanesnet.utils.gaussian import SpectralBasis, gaussian_fit
from xanesnet.utils.mode import Mode

from .registry import DatasetRegistry
from .torch_dataset import TorchDataset


@dataclass
class Data:
    x: torch.Tensor = None
    y: torch.Tensor = None
    e: torch.Tensor = None
    fourier: torch.Tensor = None
    c_star: torch.Tensor = None
    file_name: str = None

    def to(self, device):
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
        mode: Mode,
        preload: bool,
        params: dict,
        descriptors: list,
    ):
        # Calling parent init
        super().__init__(dataset_type, datasource, root, mode, preload, params)

        # Some assertions
        if self.params.get("fourier", False) or self.params.get("gaussian", False):
            if self.mode is not Mode.FORWARD:
                raise NotImplementedError("Fourier and Gaussian features are only allowed in FORWARD mode.")
            if self.params.get("fourier", False) and self.params.get("gaussian", False):
                raise NotImplementedError("Fourier and Gaussian features cannot be used together.")

        # Create descriptors
        self.descriptor_list = []
        descriptor_types = ", ".join(d["descriptor_type"] for d in descriptors)
        logging.info(f"Initialising descriptors: {descriptor_types}")
        for descriptor_config in descriptors:
            descriptor = DescriptorRegistry.get(descriptor_config["descriptor_type"])(
                **descriptor_config.get("params", {})
            )
            self.descriptor_list.append(descriptor)

        # Setup spectral basis only if needed
        if self.params.get("gaussian", False):
            self._setup_spectral_basis()

    def process(self):
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
            if self.params.get("fourier", False):
                fourier = fft(intensities.numpy(), self.params.get("fourier_concat", False))

            # Gaussian
            c_star = None
            if self.params.get("gaussian", False):
                c_star = gaussian_fit(basis=self.basis, xanes=intensities.numpy())

            # Mode
            if self.mode == Mode.FORWARD:
                x = descriptor_features
                y = intensities
            elif self.mode == Mode.REVERSE:
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

    def _setup_spectral_basis(self):
        # Load directly from file
        if self.params.get("basis_path", None) is not None:
            logging.info(f"Loading spectral basis from file @ {self.params['basis_path']}")
            self.basis = torch.load(self.params["basis_path"])
        # Create from datasource
        else:
            logging.info("Creating spectral basis from datasource")
            first_data = next(iter(self.datasource))
            # ? Does this require that every XANES spectrum has the same energy grid?
            energies = first_data.site_properties["XANES"][0]["energies"]
            self.basis = SpectralBasis(
                energies=energies,
                widths_eV=self.params.get("widths_eV", [0.5, 1.0, 2.0, 4.0]),
                normalize_atoms=True,
                stride=self.params.get("basis_stride", 2),
            )

    @property
    def metadata(self) -> dict:
        """Return dataset metadata as a dictionary."""
        metadata = super().metadata
        in_size = len(self[0].x)
        if self.params.get("gaussian", False):
            feature = self[0].c_star
        elif self.params.get("fourier", False):
            feature = self[0].fourier
        else:
            feature = self[0].y
        out_size = 0 if feature is None else len(feature)

        metadata.update({"in_size": in_size, "out_size": out_size})

        return metadata

    def collate_fn(self, batch):
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
