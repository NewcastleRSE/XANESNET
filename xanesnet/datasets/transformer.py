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
import torch.nn as nn

from pathlib import Path
from typing import List, Union
from dataclasses import dataclass
from ase.data import atomic_masses
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.registry import register_dataset
from xanesnet.utils.fourier import fft
from xanesnet.utils.io import list_filestems, load_xanes, transform_xyz, load_xyz
from xanesnet.utils.mode import Mode


@dataclass
class Data:
    mace: torch.Tensor = None
    desc: torch.Tensor = None
    pos: torch.Tensor = None
    weight: torch.Tensor = None
    mask: torch.Tensor = None
    y: torch.Tensor = None
    e: torch.Tensor = None

    def to(self, device):
        # send batch do device
        for attr in ["mace", "desc", "pos", "weight", "mask", "y", "e"]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self


@register_dataset("transformer")
class TransformerDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        xyz_path: List[str] | str = None,
        xanes_path: List[str] | str = None,
        mode: Mode = None,
        descriptors: list = None,
        **kwargs,
    ):
        # Unpack kwargs
        self.fft = kwargs.get("fourier", False)
        self.fft_concat = kwargs.get("fourier_concat", False)

        # dataset accepts only one path each for the XYZ and XANES datasets.
        xyz_path = self.unique_path(xyz_path)
        xanes_path = self.unique_path(xanes_path)

        BaseDataset.__init__(
            self, Path(root), xyz_path, xanes_path, mode, descriptors, **kwargs
        )

        if self.mode is not Mode.XYZ_TO_XANES:
            raise ValueError(f"Unsupported mode for TransformerDataset: {self.mode}")

        if not self.xyz_path:
            raise ValueError(f"Undefined xyz_path")

        # Save configuration
        params = {
            "fourier": self.fft,
            "fourier_concat": self.fft_concat,
        }
        self.register_config(locals(), type="transformer")

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
        else:
            raise ValueError("At least one data dataset path must be provided.")

        if not file_names:
            raise ValueError("No matching files found in the provided paths.")

        self.file_names = file_names

    def process(self):
        """Processes raw XYZ and XANES file to convert them into data objects."""
        # separate MACE from other descriptors
        mace_desc = next((d for d in self.descriptors if d.get_type() == "mace"), None)
        feat_desc = [d for d in self.descriptors if d.get_type() != "mace"]

        mace_list, desc_list = [], []
        spec_list, e_list = [], []
        pos_list, weight_list, mask_list = [], [], []

        for stem in tqdm(self.file_names, total=len(self.file_names)):
            if self.xyz_path:
                raw_path = os.path.join(self.xyz_path, f"{stem}.xyz")
                with open(raw_path, "r") as f:
                    atoms = load_xyz(f)

                # non-MACE descriptor feature
                desc = transform_xyz(raw_path, feat_desc)
                desc_list.append(desc)

                # MACE feature
                mace_feat = torch.tensor(
                    mace_desc.transform(atoms), dtype=torch.float32
                )
                mace_list.append(mace_feat)

                # atomic mask
                mask_list.append(torch.ones(mace_feat.shape[0], dtype=torch.bool))

                # atomic positions
                pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
                pos_list.append(pos)

                # atomic weights
                weight = torch.tensor(
                    [atomic_masses[Z] for Z in atoms.get_atomic_numbers()],
                    dtype=torch.float32,
                )
                weight_list.append(weight)

            # process xanes
            if self.xanes_path:
                raw_path = os.path.join(self.xanes_path, f"{stem}.txt")
                e, xanes = load_xanes(raw_path)
                if self.fft:
                    xanes = fft(xanes, self.fft_concat)

                spec_list.append(xanes)
                e_list.append(e)

        # normalised mace encoding
        mace_tensor = torch.cat(mace_list, dim=0)
        mean = mace_tensor.mean(dim=0, keepdim=True)
        std = mace_tensor.std(dim=0, keepdim=True) + 1e-8
        mace_norm_list = [(mdd - mean) / std for mdd in mace_list]

        for idx, stem in tqdm(enumerate(self.file_names), total=len(self.file_names)):
            data = Data(
                mace=mace_norm_list[idx],
                desc=desc_list[idx],
                pos=pos_list[idx],
                weight=weight_list[idx],
                mask=mask_list[idx],
                y=spec_list[idx] if spec_list else None,
                e=e_list[idx] if e_list else None,
            )

            save_path = os.path.join(self.processed_dir, f"{stem}.pt")
            torch.save(data, save_path)

    def collate_fn(self, batch: list[Data]) -> Data:
        """
        Collates a list of Data objects into a single Data object with batched tensors.
        """
        mace_list = [sample.mace for sample in batch]
        desc_list = [sample.desc for sample in batch]
        pos_list = [sample.pos for sample in batch]
        weight_list = [sample.weight for sample in batch]
        mask_list = [sample.mask for sample in batch]
        spec_list = [sample.y for sample in batch]

        desc = torch.stack(desc_list).to(torch.float32)
        spec = torch.stack(spec_list).to(torch.float32)

        mace = pad_sequence(mace_list, batch_first=True).to(torch.float32)
        weight = pad_sequence(weight_list, batch_first=True).to(torch.float32)
        pos = pad_sequence(pos_list, batch_first=True).to(torch.float32)
        mask = pad_sequence(mask_list, batch_first=True).to(torch.bool)

        return Data(mace=mace, desc=desc, pos=pos, weight=weight, mask=mask, y=spec)

    @property
    def x_size(self) -> Union[int, List[int]]:
        """Size of the feature array."""
        return [self[0].mace.shape[1], self[0].desc.shape[0]]

    @property
    def y_size(self) -> int:
        """Size of the label array."""
        return self[0].y.shape[0]
