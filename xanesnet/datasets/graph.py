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
from pathlib import Path

import torch
import numpy as np

from tqdm import tqdm
from typing import List, Union
from torch_geometric.data import Data, Dataset, Batch

from xanesnet.core_learn import Mode
from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.registry import register_dataset
from xanesnet.utils.fourier import fft
from xanesnet.utils.io import list_filestems, load_xanes
from xanesnet.utils.xyz2graph import MolGraph


@register_dataset("graph")
class GraphDataset(BaseDataset):
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
        self.n = kwargs.get("n", 16)
        self.r_min = kwargs.get("r_min", 0.0)
        self.r_max = kwargs.get("r_max", 4.0)
        self.fft = kwargs.get("fourier", False)
        self.fft_concat = kwargs.get("fourier_concat", False)

        # dataset accepts only one path each for the XYZ and XANES datasets.
        xyz_path = self.unique_path(xyz_path)
        xanes_path = self.unique_path(xanes_path)

        BaseDataset.__init__(
            self, Path(root), xyz_path, xanes_path, mode, descriptors, **kwargs
        )

        if self.mode is not Mode.XYZ_TO_XANES:
            raise ValueError(f"Unsupported mode for GraphDataset: {self.mode}")

        # Save configuration
        params = {
            "fourier": self.fft,
            "fourier_concat": self.fft_concat,
            "n": self.n,
            "r_min": self.r_min,
            "r_max": self.r_max,
        }
        self.register_config(locals(), type="graph")

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

    def collate_fn(self, batch):
        """Custom collate function to handle a list of Data objects."""
        # This will be handle in torch_geometric.data.DataLoader
        return None

    @property
    def x_size(self) -> Union[int, List[int]]:
        # node feature size and graph attribute size
        return [self[0].x.shape[1], self[0].graph_attr.shape[0]]

    @property
    def y_size(self) -> Union[int, List[int]]:
        # xanes (label) size
        return len(self[0].y)

    def process(self):
        """
        Processes raw XYZ and XANES files to convert them into graph data objects.
        """
        for idx, stem in tqdm(enumerate(self.file_names), total=len(self.file_names)):
            # Get energy and intensities
            xanes = e = None
            if self.xanes_path:
                raw_path = os.path.join(self.xanes_path, f"{stem}.txt")
                e, xanes = load_xanes(raw_path)
                if self.fft:
                    xanes = fft(xanes, self.fft_concat)

            mg = MolGraph()
            raw_path = os.path.join(self.xyz_path, f"{stem}.xyz")
            mg.read_xyz(raw_path)

            data = Data(
                x=self._get_node_features(mg),
                edge_index=mg.edge_index,
                edge_attr=self._get_edge_features(mg),
                y=xanes,
                graph_attr=self._get_graph_features(mg),
                name=stem,
                e=e,
            )

            save_path = os.path.join(self.processed_dir, f"{stem}.pt")
            torch.save(data, save_path)

    def _get_node_features(self, mg: MolGraph):
        """
        Return a 2d array of the shape [Number of Nodes, Node Feature size]
        """

        atomic_numbers = list(range(2, 21))

        mole_atomic_numbers = mg.atoms.get_atomic_numbers()

        one_hot_encoding = np.zeros(
            (len(mole_atomic_numbers), len(atomic_numbers) + 1), dtype=int
        )
        # Set one_hot_encoding for absorber
        one_hot_encoding[0, 0] = 1

        # Set one_hot_encoding for atomic number
        for i, atomic_number in enumerate(mole_atomic_numbers):
            if atomic_number in atomic_numbers:
                index = atomic_numbers.index(atomic_number)
                one_hot_encoding[i, index] = 1

        all_node_feats = one_hot_encoding

        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mg: MolGraph):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """

        num_edges = len(mg.edge_list)
        all_edge_feats = np.full((num_edges, self.n), np.nan)

        r_aux = np.linspace(self.r_min + 0.5, self.r_max - 0.5, self.n)
        dr = np.diff(r_aux)[0]
        width = np.array([1.0 / (2.0 * (dr**2)) for _ in r_aux])
        grid = np.array([i for i in r_aux])
        bond_lengths = np.array([mg.bond_lengths[i] for i in mg.edge_list])
        cutoffs = (np.cos((np.pi * bond_lengths) / self.r_max) + 1.0) / 2.0

        for i in range(num_edges):
            g2 = gaussian(bond_lengths[i], width, grid)
            all_edge_feats[i, :] = np.sum(g2 * cutoffs[i], axis=0)

        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_graph_features(self, mg: MolGraph):
        """
        This will return 1d vector of the shape
        [Feature size]
        """
        n_feats = 0
        # Feature array pre-allocation
        for descriptor in self.descriptors:
            n_feats += descriptor.get_nfeatures()
        all_graph_feats = np.full(n_feats, np.nan)

        s = 0
        for descriptor in self.descriptors:
            l = descriptor.get_nfeatures()
            all_graph_feats[s : s + l] = descriptor.transform(mg.atoms)
            s += l

        return torch.tensor(all_graph_feats, dtype=torch.float)


def gaussian(r: np.ndarray, h: float, m: float) -> np.ndarray:
    """returns a gaussian-like function defined over `r`"""
    return np.exp(-1.0 * h * (r - m) ** 2)
