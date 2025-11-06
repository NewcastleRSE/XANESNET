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

import math
import os
import numpy as np
import torch

from pathlib import Path
from typing import List, Union
from dataclasses import dataclass

from ase import Atoms
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.models.softshell import SpectralBasis
from xanesnet.registry import register_dataset
from xanesnet.utils.io import list_filestems, load_xanes, transform_xyz, load_xyz
from xanesnet.utils.mode import Mode


@dataclass
class Data:
    desc: torch.Tensor = None  # descriptor feature
    dist: torch.Tensor = None  # distance feature
    y: torch.Tensor = None  # label (spectra)
    e: torch.Tensor = None  # energies
    c_star: torch.Tensor = None  # coefficient C*
    lengths: torch.Tensor = None

    def to(self, device):
        # send batch do device, e is excluded
        for attr in ["desc", "dist", "y", "c_star", "lengths"]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self


@register_dataset("softshell")
class SoftShellDataset(BaseDataset):
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
        self.basis_stride = kwargs.get("basis_stride", 4)

        # dataset accepts only one path each for the XYZ and XANES datasets.
        xyz_path = self.unique_path(xyz_path)
        xanes_path = self.unique_path(xanes_path)

        BaseDataset.__init__(
            self, Path(root), xyz_path, xanes_path, mode, descriptors, **kwargs
        )

        if self.mode is not Mode.XYZ_TO_XANES:
            raise ValueError(f"Unsupported mode for SoftShellDataset: {self.mode}")

        if not self.xyz_path:
            raise ValueError(f"Undefined xyz_path")

        # Save configuration
        params = {
            "fourier": self.fft,
            "fourier_concat": self.fft_concat,
            "basis_stride": self.basis_stride,
        }
        self.register_config(locals(), type="softshell")

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
        energy_flag = 0
        y_len = 0
        A = None

        for idx, stem in tqdm(enumerate(self.file_names), total=len(self.file_names)):
            raw_path = os.path.join(self.xyz_path, f"{stem}.xyz")
            # descriptor feature tensor
            desc = transform_xyz(raw_path, self.descriptors)

            with open(raw_path, "r") as f:
                mol = load_xyz(f)

            # distance feature tensor
            dist = self.distances_to_absorber(mol, absorber_idx=0)  # (n_atoms,)

            # energy (eV) intensities (xanes), and energy coeffs (c_star) tensors
            eV = c_star = None
            if self.xanes_path:
                raw_path = os.path.join(self.xanes_path, f"{stem}.txt")
                eV, xanes = load_xanes(raw_path)

                dE = float(eV[1] - eV[0])
                torch.set_printoptions(precision=8)

                if not energy_flag:
                    widths_eV = (0.5, 1.0, 2.0, 4.0)
                    widths_bins = tuple(max(w / dE, 0.5) for w in widths_eV)
                    y_len = len(xanes)

                    basis = SpectralBasis(
                        energies=eV,
                        widths_bins=widths_bins,
                        normalize_atoms=True,
                        stride=self.basis_stride,
                    )

                    # Ridge operator for Φ and coefficients c*
                    RIDGE_LAMBDA = 1e-2
                    A = self.build_ridge_operator(basis.Phi, lam=RIDGE_LAMBDA)  # (K, N)

                    energy_flag = 1

                if len(xanes) != y_len:
                    raise ValueError(
                        f"Spectrum length mismatch for {stem}: expected {y_len}, got {len(xanes)}."
                    )

                c_star = xanes @ A.T

            # initialise data object
            data = Data(desc=desc, dist=dist, y=xanes, e=eV, c_star=c_star)
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

        batched_desc = pad_sequence(desc_list, batch_first=True).to(torch.float32)
        batched_dist = pad_sequence(dist_list, batch_first=True).to(torch.float32)
        batched_y = torch.stack(y_list, dim=0).to(torch.float32)
        batched_c = torch.stack(c_list, dim=0).to(torch.float32)

        return Data(
            desc=batched_desc,
            dist=batched_dist,
            y=batched_y,
            c_star=batched_c,
            lengths=lengths,
        )

    @property
    def x_size(self) -> Union[int, List[int]]:
        """Size of the feature array."""
        x_size = []
        eV = self[0].e

        dE = eV[1] - eV[0]
        widths_eV = (0.5, 1.0, 2.0, 4.0)
        widths_bins = tuple(max(w / dE, 0.5) for w in widths_eV)

        basis = SpectralBasis(
            energies=eV,
            widths_bins=widths_bins,
            normalize_atoms=True,
            stride=self.basis_stride,
        )

        # Per-width group sizes for grouped head
        K = basis.Phi.shape[1]
        n_width_groups = len(widths_bins)

        # Number of centers per width (should be equal for each width given same stride)
        per_width = K // n_width_groups
        K_groups = [per_width] * n_width_groups

        # Sanity: sum of groups equals K
        assert sum(K_groups) == K, f"K_groups {K_groups} do not sum to K={K}"

        # Append descriptor feature size and K_groups to x_size
        x_size.append(self[0].desc.shape[1])
        x_size.append(K_groups)

        return x_size

    @property
    def y_size(self) -> int:
        """Size of the label array."""
        return self[0].y.shape[0]

    @staticmethod
    def distances_to_absorber(mol: Atoms, absorber_idx: int = 0) -> Tensor:
        pos = mol.get_positions()
        ref = pos[absorber_idx]
        d = np.linalg.norm(pos - ref, axis=1).astype(np.float32)
        return torch.tensor(d, dtype=torch.float32)

    @staticmethod
    def normalize_area(
        xanes: Tensor,
        delta_e: float,
        target_area: float = 1.0,
        baseline: str = "shift_nonneg",
        eps: float = 1e-12,
    ) -> Tensor:
        y = xanes.clone().float()

        if baseline == "shift_nonneg":
            m = y.min()
            if m < 0:
                y = y - m
        area = float(y.sum()) * float(delta_e)

        if area <= eps:
            return torch.zeros_like(y, dtype=torch.float32)
        scale = target_area / area

        return y * scale

    @staticmethod
    def build_ridge_operator(Phi: Tensor, lam: float = 1e-2) -> Tensor:
        """
        A = (Φᵀ Φ + λ I)^{-1} Φᵀ  with Cholesky; fallback to augmented LSQ.
        Returns A: (K, N_E) on same device/dtype as Phi.
        """
        Phi = Phi.contiguous()
        N_E, K = Phi.shape
        I_K = torch.eye(K, dtype=Phi.dtype, device=Phi.device)

        G = Phi.T @ Phi
        G = G + lam * I_K
        try:
            L = torch.linalg.cholesky(G)  # (K,K)
            A = torch.cholesky_solve(Phi.T, L)  # (K,N_E)
        except RuntimeError:
            top = Phi
            bot = math.sqrt(lam) * I_K
            A_aug = torch.cat([top, bot], dim=0)  # ((N_E+K), K)
            rhs = torch.cat(
                [
                    torch.eye(N_E, dtype=Phi.dtype, device=Phi.device),
                    torch.zeros((K, N_E), dtype=Phi.dtype, device=Phi.device),
                ],
                dim=0,
            )  # ((N_E+K), N_E)
            A = torch.linalg.lstsq(A_aug, rhs, rcond=None).solution  # (K, N_E)

        return A.to(torch.float32)
