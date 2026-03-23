"""
XANESNET Energy-Embedded E3NN dataset for absorber-centred e3nn models.

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
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ase.data import atomic_numbers
from ase.io import read

from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.registry import register_dataset
from xanesnet.utils.io import list_filestems
from xanesnet.utils.mode import Mode

@dataclass
class GraphData:
    z: Optional[torch.Tensor] = None        
    pos: Optional[torch.Tensor] = None      
    mask: Optional[torch.Tensor] = None     
    y: Optional[torch.Tensor] = None        
    e: Optional[torch.Tensor] = None        
    absorber_index: Optional[torch.Tensor] = None  
    stem: Optional[str] = None

    def to(self, device):
        for attr in ["z", "pos", "mask", "y", "e", "absorber_index"]:
            val = getattr(self, attr)
            if val is not None and torch.is_tensor(val):
                setattr(self, attr, val.to(device))
        return self


@register_dataset("e3eembed")
class E3EEmbdedDataset(BaseDataset):

    def __init__(
        self,
        root: str,
        xyz_path: List[str] | str = None,
        xanes_path: List[str] | str = None,
        mode: Mode = None,
        descriptors: list = None,
        **kwargs,
    ):
        xyz_path = self._unique_path(xyz_path)
        xanes_path = self._unique_path(xanes_path)

        # dataset-specific kwargs
        self.absorber_index = kwargs.get("absorber_index", 0)

        super().__init__(
            Path(root),
            xyz_path,
            xanes_path,
            mode,
            descriptors,
            **kwargs,
        )

        self._register_config(
            dataset_type="e3eembed",
            absorber_index=self.absorber_index,
        )

    def set_file_names(self):
        xyz_path = self.xyz_path
        xanes_path = self.xanes_path

        if xyz_path and xanes_path:
            xyz_stems = set(list_filestems(xyz_path))
            xanes_stems = set(list_filestems(xanes_path))
            file_names = sorted(list(xyz_stems & xanes_stems))
        elif xyz_path:
            file_names = sorted(list(set(list_filestems(xyz_path))))
        elif xanes_path:
            file_names = sorted(list(set(list_filestems(xanes_path))))
        else:
            raise ValueError("At least one of xyz_path or xanes_path must be provided.")

        if not file_names:
            raise ValueError("No matching files found in the provided paths.")

        self.file_names = file_names

    def process(self):
        for stem in tqdm(self.file_names, total=len(self.file_names)):
            z = pos = mask = y = e = None

            if self.xyz_path:
                xyz_file = os.path.join(self.xyz_path, f"{stem}.xyz")
                atoms = read(xyz_file)
                z_np = atoms.numbers.astype(np.int64)
                pos_np = atoms.positions.astype(np.float32)
                z = torch.tensor(z_np, dtype=torch.long)
                pos = torch.tensor(pos_np, dtype=torch.float32)
                mask = torch.ones(len(z_np), dtype=torch.bool)

            if self.xanes_path:
                xanes_file = os.path.join(self.xanes_path, f"{stem}.txt")
                e, xanes = self.transform_xanes(xanes_file)

            if self.mode == Mode.XANES_TO_XYZ:
                raise NotImplementedError(
                    "e3eembed is intended for XYZ -> XANES forward models."
                )
            else:
                y = xanes

            data = GraphData(
                z=z,
                pos=pos,
                mask=mask,
                y=y,
                e=e,
                absorber_index=torch.tensor(self.absorber_index, dtype=torch.long),
                stem=stem,
            )
            torch.save(data, os.path.join(self.processed_dir, f"{stem}.pt"))

    def collate_fn(self, batch: list[GraphData]) -> GraphData:
        z_list = [sample.z for sample in batch]
        pos_list = [sample.pos for sample in batch]
        mask_list = [sample.mask for sample in batch]
        y_list = [sample.y for sample in batch]
        e_list = [sample.e for sample in batch]
        stem_list = [getattr(sample, "stem", None) for sample in batch]


        z = self._safe_pad(z_list, dtype=torch.long)
        pos = self._safe_pad(pos_list, dtype=torch.float32)
        mask = self._safe_pad(mask_list, dtype=torch.bool)
        y = self._safe_stack(y_list, dtype=torch.float32)

        e = e_list[0] if all(x is not None for x in e_list) else None

        data = GraphData(
            z=z,
            pos=pos,
            mask=mask,
            y=y,
            e=e,
            absorber_index=torch.tensor(self.absorber_index, dtype=torch.long),
            stem=stem_list,
        )
        return data

    @property
    def in_features(self):
        return 1

    @property
    def out_features(self):
        y = self[0].y
        return 0 if y is None else len(y)
