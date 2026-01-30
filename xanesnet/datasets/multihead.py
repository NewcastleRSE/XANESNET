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
from dataclasses import dataclass

import torch

from pathlib import Path
from typing import List
from tqdm import tqdm

from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.registry import register_dataset
from xanesnet.utils.io import list_filestems
from xanesnet.utils.mode import Mode


@dataclass
class Data:
    x: torch.Tensor = None
    y: torch.Tensor = None
    e: torch.Tensor = None
    head_idx: torch.Tensor = None  # multihead index
    head_name: str = ""

    def to(self, device):
        # send batch do device
        for attr in ["x", "y", "head_idx"]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self


@register_dataset("multihead")
class MultiheadDataset(BaseDataset):
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
        xyz_path = self._list_path(xyz_path)
        xanes_path = self._list_path(xanes_path)
        head_names = [p.name for p in xanes_path]

        BaseDataset.__init__(
            self, Path(root), xyz_path, xanes_path, mode, descriptors, **kwargs
        )

        # Save configuration
        self._register_config(dataset_type="multihead", head_names=head_names)

    def set_file_names(self):
        """
        Get the list of valid file stems based on the
        xyz_path and/or xanes_path. If both are given, only common stems are kept.
        """
        xyz_path = self.xyz_path
        xanes_path = self.xanes_path

        if xyz_path and xanes_path:
            xyz_stems = set(stem for d in xyz_path for stem in list_filestems(d))
            xanes_stems = set(stem for d in xanes_path for stem in list_filestems(d))
            # The intersection logic remains the same
            file_names = sorted(list(xyz_stems & xanes_stems))
        elif xyz_path:
            xyz_stems = set(stem for d in xyz_path for stem in list_filestems(d))
            file_names = sorted(list(xyz_stems))
        elif xanes_path:
            xanes_stems = set(stem for d in xanes_path for stem in list_filestems(d))
            file_names = sorted(list(xanes_stems))
        else:
            raise ValueError("At least one data path must be provided.")

        if not file_names:
            raise ValueError("No matching files found in the provided paths.")

        self.file_names = file_names

    def process(self):
        """Processes raw XYZ and XANES file to convert them into data objects."""
        for idx, stem in tqdm(enumerate(self.file_names), total=len(self.file_names)):
            xyz = xanes = e = None
            head_idx_xyz = head_idx_xanes = None
            head_name_xyz = head_name_xanes = None

            # transform xyz file into feature array
            if self.xyz_path:
                xyz_file, head_idx_xyz, head_name_xyz = self.find_file(
                    stem, self.xyz_path, ".xyz"
                )
                xyz = self.transform_xyz(xyz_file)

            # get xanes, energy, head index and name
            if self.xanes_path:
                xanes_file, head_idx_xanes, head_name_xanes = self.find_file(
                    stem, self.xanes_path, ".txt"
                )
                e, xanes = self.transform_xanes(xanes_file)

            if self.mode == Mode.XANES_TO_XYZ:
                x = xanes
                y = xyz
                head_idx = head_idx_xyz
                head_name = head_name_xyz
            else:
                x = xyz
                y = xanes
                head_idx = head_idx_xanes
                head_name = head_name_xanes

            if head_idx is not None:
                head_idx = torch.tensor(head_idx, dtype=torch.long)

            data = Data(x=x, y=y, e=e, head_idx=head_idx, head_name=head_name)

            save_path = os.path.join(self.processed_dir, f"{stem}.pt")
            torch.save(data, save_path)

    def find_file(self, stem, dirs, ext):
        """
        Return the first matching file in dirs
        """
        for idx, d in enumerate(dirs):
            file = os.path.join(d, f"{stem}{ext}")
            if os.path.exists(file):
                parent_dir = os.path.basename(os.path.dirname(file))
                return file, idx, parent_dir
        raise ValueError(f"Cannot find matching file: {stem}.")

    def collate_fn(self, batch: list[Data]) -> Data:
        """
        Collates a list of Data objects into a single Data object  with batched tensors.
        """
        x_list = [sample.x for sample in batch]
        y_list = [sample.y for sample in batch]
        idx_list = [sample.head_idx for sample in batch]

        batched_x = self._safe_stack(x_list)
        batched_y = self._safe_stack(y_list)
        batched_i = self._safe_stack(idx_list, dtype=torch.long)

        return Data(
            x=batched_x,
            y=batched_y,
            head_idx=batched_i,
        )

    @property
    def in_features(self) -> List[int] | int:
        """Size of the feature array."""
        return len(self[0].x)

    @property
    def out_features(self) -> List[int] | int:
        """Size of the label array."""
        # mapping each head index to the length of its label array.
        if self[0].y is None:
            return 0
        out_sizes = {int(data.head_idx): len(data.y) for data in self}

        # return the dict values as a list
        return [int(v) for _, v in sorted(out_sizes.items())]
