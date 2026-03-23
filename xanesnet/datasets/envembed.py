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
import numpy as np
import torch

from pathlib import Path
from typing import List, Union
from dataclasses import dataclass

from ase import Atoms
from torch import Tensor
from tqdm import tqdm

from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.registry import register_dataset
from xanesnet.utils.io import list_filestems, load_xanes, load_xyz
from xanesnet.utils.mode import Mode
from xanesnet.utils.gaussian import gaussian_forward


@dataclass
class Data:
    desc: torch.Tensor = None  # descriptor feature
    dist: torch.Tensor = None  # distance feature
    y: torch.Tensor = None  # label (spectra)
    e: torch.Tensor = None  # energies
    c_star: torch.Tensor = None  # coefficient C*
    lengths: torch.Tensor = None
    stem: Union[str, List[str]] = None  # filename stem

    def to(self, device):
        # send batch do device, e is excluded
        for attr in ["desc", "dist", "y", "c_star", "lengths"]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self


@register_dataset("envembed")
class EnvEmbedDataset(BaseDataset):
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
        xyz_path = self._unique_path(xyz_path)
        xanes_path = self._unique_path(xanes_path)

        BaseDataset.__init__(
            self, Path(root), xyz_path, xanes_path, mode, descriptors, **kwargs
        )

        if self.mode is not Mode.XYZ_TO_XANES:
            raise ValueError(f"Unsupported mode for EnvEmbedDataset: {self.mode}")

        if not self.xyz_path:
            raise ValueError(f"Undefined xyz_path")

        # Save configuration
        self._register_config(dataset_type="envembed")

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
        for idx, stem in tqdm(enumerate(self.file_names), total=len(self.file_names)):
            xyz_file = os.path.join(self.xyz_path, f"{stem}.xyz")
            desc = self.transform_xyz(xyz_file)

            with open(xyz_file, "r") as f:
                mol = load_xyz(f)

            # distance feature tensor
            dist = self.distances_to_absorber(mol, absorber_idx=0)  # (n_atoms,)

            e = c_star = xanes = None
            if self.xanes_path:
                xanes_file = os.path.join(self.xanes_path, f"{stem}.txt")
                e, xanes = load_xanes(xanes_file)

                c_star = gaussian_forward(basis=self.gauss_basis, xanes=xanes)

            # initialise data object
            data = Data(desc=desc, dist=dist, y=xanes, e=e, c_star=c_star, stem=stem)
            # save data to disk
            save_path = os.path.join(self.processed_dir, f"{stem}.pt")
            torch.save(data, save_path)

    def collate_fn(self, batch: list[Data]) -> Data:
        """
        Collates a list of Data objects into a single Data object with batched tensors.
        """
        desc_list = [sample.desc for sample in batch]
        dist_list = [sample.dist for sample in batch]
        y_list = [sample.y for sample in batch]
        c_list = [sample.c_star for sample in batch]
        lengths = torch.tensor([d.size(0) for d in desc_list], dtype=torch.long)
        stem_list = [sample.stem for sample in batch]   # <-- added


        batched_desc = self._safe_pad(desc_list)
        batched_dist = self._safe_pad(dist_list)
        batched_y = self._safe_stack(y_list)
        batched_c = self._safe_stack(c_list)

        return Data(
            desc=batched_desc,
            dist=batched_dist,
            y=batched_y,
            c_star=batched_c,
            lengths=lengths,
            stem=stem_list,   # <-- added
        )

    @property
    def in_features(self) -> List[int] | int:
        """Size of the feature array."""
        x_size = []
        e = self.gauss_basis.E

        # Per-width group sizes for grouped head
        dE = e[1] - e[0]
        widths_bins = tuple(max(w / dE, 0.5) for w in self.widths_eV)
        n_width_groups = len(widths_bins)

        # Number of centers per width (should be equal for each width given same stride)
        K = self.gauss_basis.Phi.shape[1]
        per_width = K // n_width_groups
        K_groups = [per_width] * n_width_groups

        # Sanity: sum of groups equals K
        assert sum(K_groups) == K, f"K_groups {K_groups} do not sum to K={K}"

        # Append descriptor feature size and K_groups to x_size
        x_size.append(self[0].desc.shape[1])
        x_size.append(K_groups)

        return x_size

    @property
    def out_features(self) -> List[int] | int:
        """Size of the label array."""
        y = self[0].y
        return 0 if y is None else y.shape[0]

    @staticmethod
    def distances_to_absorber(mol: Atoms, absorber_idx: int = 0) -> Tensor:
        pos = mol.get_positions()
        ref = pos[absorber_idx]
        d = np.linalg.norm(pos - ref, axis=1).astype(np.float32)
        return torch.tensor(d, dtype=torch.float32)
