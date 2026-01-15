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

import os
import torch

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.registry import register_dataset
from xanesnet.utils.fourier import fft
from xanesnet.utils.io import list_filestems, load_xanes, transform_xyz
from xanesnet.utils.mode import Mode
from xanesnet.utils.gaussian import gaussian_fit


@dataclass
class Data:
    x: torch.Tensor = None
    y: torch.Tensor = None
    e: torch.Tensor = None
    fourier: torch.Tensor = None
    c_star: torch.Tensor = None

    def to(self, device):
        # send batch do device
        for attr in ["x", "y", "fourier", "c_star"]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self


@register_dataset("xanesx")
class XanesXDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        xyz_path: List[str] | str = None,
        xanes_path: List[str] | str = None,
        mode: Mode = None,
        descriptors: list = None,
        **kwargs,
    ):
        # dataset accepts only one path each for the XYZ and XANES datasets.
        xyz_path = self.unique_path(xyz_path)
        xanes_path = self.unique_path(xanes_path)

        BaseDataset.__init__(
            self, Path(root), xyz_path, xanes_path, mode, descriptors, **kwargs
        )

        # Save configuration
        self.register_config(locals(), type="xanesx")

    def set_file_names(self):
        """
        Get the list of valid file stems based on the
        xyz_path and/or xanes_path. If both are given, only common stems are kept.
        """
        xyz_path = self.xyz_path
        xanes_path = self.xanes_path

        if xyz_path and xanes_path:
            xyz_stems = set(list_filestems(xyz_path))
            xanes_stems = set(list_filestems(xanes_path))
            file_names = sorted(list(xyz_stems & xanes_stems))
        elif xyz_path:
            xyz_stems = set(list_filestems(xyz_path))
            file_names = sorted(list(xyz_stems))
        elif xanes_path:
            xanes_stems = set(list_filestems(xanes_path))
            file_names = sorted(list(xanes_stems))
        else:
            raise ValueError("At least one data dataset path must be provided.")

        if not file_names:
            raise ValueError("No matching files found in the provided paths.")

        self.file_names = file_names

    def process(self):
        """Processes raw XYZ and XANES file to convert them into data objects."""

        for idx, stem in tqdm(enumerate(self.file_names), total=len(self.file_names)):
            xyz = xanes = e = fourier = c_star = None

            # transform xyz file into feature array
            if self.xyz_path:
                raw_path = os.path.join(self.xyz_path, f"{stem}.xyz")
                xyz = transform_xyz(raw_path, self.descriptors)

            # get xanes and energy arrays
            if self.xanes_path:
                raw_path = os.path.join(self.xanes_path, f"{stem}.txt")
                e, xanes = load_xanes(raw_path)

                if self.fft:
                    fourier = fft(xanes)
                elif self.gaussian:
                    c_star = gaussian_fit(basis=self.gauss_basis, xanes=xanes)

            if self.mode == Mode.XANES_TO_XYZ:
                x = xanes
                y = xyz
            else:
                x = xyz
                y = xanes

            data = Data(x=x, y=y, e=e, fourier=fourier, c_star=c_star)

            save_path = os.path.join(self.processed_dir, f"{stem}.pt")
            torch.save(data, save_path)

    def collate_fn(self, batch: list[Data]) -> Data:
        """
        Collates a list of Data objects into a single Data object  with batched tensors.
        """
        keys = ["x", "y", "fourier", "c_star"]
        batched = {}

        for k in keys:
            lst = [getattr(sample, k) for sample in batch]
            batched[k] = self.safe_stack(lst)

        return Data(**batched)

    @property
    def x_size(self) -> Union[int, List[int]]:
        """Size of the feature array."""
        return len(self[0].x)

    @property
    def y_size(self) -> Union[int, List[int]]:
        """Size of the label array."""
        if self.gaussian:
            c_star = self[0].c_star
            return 0 if c_star is None else len(c_star)
        elif self.fft:
            fourier = self[0].fourier
            return 0 if fourier is None else len(fourier)
        else:
            y = self[0].y
            return 0 if y is None else len(y)
