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

import copy
import logging
import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from collections.abc import Sequence
from pathlib import Path
from typing import Union, List, Any, Callable
from torch import Tensor
from torch_geometric.io import fs

from xanesnet.utils.gaussian import GaussianBasis
from xanesnet.utils.io import load_xanes
from xanesnet.utils.mode import Mode

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class BaseDataset(Dataset):
    """Abstract base class for XANESNET datasets."""

    def __init__(
        self,
        root: Path,
        xyz_path: List[Path] | Path = None,
        xanes_path: List[Path] | Path = None,
        mode: Mode = None,
        descriptors: List = None,
        **kwargs,
    ):
        """
        Args:
            root: Root directory for dataset storage.
            xyz_path: Paths to atomic coordinate files (.xyz).
            xanes_path: Paths to XANES spectra files.
            mode: training mode.
            descriptors : List of feature descriptors.
            **kwargs: Additional optional keyword arguments.
        """

        super().__init__()

        self.root = root
        self.xyz_path = xyz_path
        self.xanes_path = xanes_path
        self.mode = mode
        self.descriptors = descriptors
        self.gauss_basis = None

        # Unpack kwargs
        self.preload = kwargs.get("preload", True)
        self.fft = kwargs.get("fourier", False)
        self.gaussian = kwargs.get("gaussian", False)
        self.widths_eV = kwargs.get("widths_eV", [0.5, 1.0, 2.0, 4.0])
        self.basis_stride = kwargs.get("basis_stride", 2)
        self.basis_path = kwargs.get("basis_path", None)

        if self.fft or self.gaussian:
            if mode is not Mode.XYZ_TO_XANES:
                raise ValueError(
                    "FFT and Gaussian transformation are only supported in XYZ_TO_XANES mode."
                )
        if self.fft and self.gaussian:
            raise ValueError(
                "FFT and Gaussian transformations cannot be applied at the same time"
            )

        self.config = {}
        self.preload_dataset = []
        self.file_names = None

        self.setup_gaussian_basis()
        self.set_file_names()
        self._process()

        params = {
            "gaussian": self.gaussian,
            "widths_eV": self.widths_eV,
            "basis_stride": self.basis_stride,
            "fourier": self.fft,
        }
        self.register_config(locals())

    def set_file_names(self):
        """Set a list of file names (stems) in the dataset."""
        raise NotImplementedError

    def process(self):
        """Process the raw file and save them to the self.processed_dir folder."""
        raise NotImplementedError

    def collate_fn(self, batch):
        """Custom collate function to handle batching of data objects."""
        raise NotImplementedError

    @property
    def x_size(self) -> Union[int, List[int]]:
        """Size or shape of the input feature tensor."""
        raise NotImplementedError

    @property
    def y_size(self) -> Union[int, List[int]]:
        """Size or shape of the target label tensor."""
        raise NotImplementedError

    @property
    def indices(self) -> Sequence:
        """List of integer indices corresponding to data entries."""
        return list(range(len(self.file_names)))

    @property
    def processed_file_names(self) -> List[str]:
        """List of processed data file names."""
        return [f"{stem}.pt" for i, stem in enumerate(self.file_names)]

    @property
    def processed_dir(self) -> str:
        """Directory path where processed data files are stored."""
        return os.path.join(self.root, "processed")

    @property
    def processed_paths(self) -> List[str]:
        """List of absolute paths to all processed data files."""
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [os.path.join(self.processed_dir, f) for f in self.to_list(files)]

    def __len__(self) -> int:
        """Number of data object in the dataset."""
        return len(self.file_names)

    def __getitem__(self, idx: Union[int, np.integer, IndexType]):
        """Retrieve a data object or a subset of the dataset by index.

        Supports integer indexing, slicing, or index arrays/tensors.

        Args:
            idx: The index or indices of items to retrieve.

        Returns:
            If `idx` is a single integer: a single data object.
            Otherwise: a subset of the dataset.
        """
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            if self.preload:
                return self.preload_dataset[self.indices[idx]]
            else:
                return torch.load(self.processed_paths[idx])

        else:
            return self.index_select(idx)

    def _process(self):
        """
        Process raw data file. If processed files exist, skip processing.
        """
        if self.files_exist(self.processed_paths):
            logging.info(
                f">> Processed files exist in {self.processed_dir}, skipping data processing."
            )
        else:
            logging.info(f"Processing {len(self.file_names)} files to data objects...")
            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()

        if self.preload:
            logging.info(">> Preloading dataset into memory...")
            self.preload_dataset = [torch.load(path) for path in self.processed_paths]

    def index_select(self, idx: IndexType) -> "BaseDataset":
        """Creates a subset of the dataset from specified indices.
        Indices can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        """
        index = self.file_names

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            index = index[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            index = [index[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        dataset = copy.copy(self)
        dataset.file_names = index
        return dataset

    def shuffle(self) -> "BaseDataset":
        """Randomly shuffles the examples in the dataset."""
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return dataset

    def register_config(self, args, **kwargs):
        """
        Assign arguments from the child class's constructor to self.config.

        Args:
            args: The dictionary of arguments from the child class's constructor
            **kwargs: additional arguments to store
        """
        selected = {k: args[k] for k in ("type", "params") if k in args}

        self.config.update(kwargs)
        self.config.update(selected)

    def setup_gaussian_basis(self):
        if self.basis_path is not None:
            logging.info(f">> Loading Gaussian basis from {self.basis_path}")
            self.gauss_basis = torch.load(self.basis_path)
            return

        if self.xanes_path:
            if isinstance(self.xanes_path, List):
                path = self.xanes_path[0]
            else:
                path = self.xanes_path

            files = sorted(Path(path).glob("*.txt"))

            if files:
                e, xanes = load_xanes(str(files[0]))
            else:
                raise ValueError(f"No XANES files were found in {path}.")

            self.gauss_basis = GaussianBasis(
                energies=e,
                widths_eV=self.widths_eV,
                normalize_atoms=True,
                stride=self.basis_stride,
            )
        else:
            raise ValueError("XANES path must be provided to set up Gaussian basis.")

    @staticmethod
    def unique_path(path) -> Path:
        """
        Resolve single path
        """
        if isinstance(path, list):
            if len(path) > 1:
                raise ValueError(
                    "Dataset does not support multiple paths. Please provide only one."
                )
            path = path[0] if path else None

        return Path(path) if path is not None else None

    @staticmethod
    def list_path(path) -> List[Path]:
        """
        Resolve list of paths
        """
        if path is None:
            return []

        if isinstance(path, list):
            return [Path(p) for p in path]

        return [Path(path)]

    @staticmethod
    def files_exist(files: List[str]) -> bool:
        """
        Check whether all given file paths exist.
        """
        return len(files) != 0 and all([fs.exists(f) for f in files])

    @staticmethod
    def to_list(value: Any) -> Sequence:
        """
        Ensure the input is returned as a list.
        """
        if isinstance(value, Sequence) and not isinstance(value, str):
            return value
        else:
            return [value]

    @staticmethod
    def safe_stack(lst, dtype=torch.float32):
        """
        Stacks tensor list.
        Returns None if any element in the list is None.
        """
        if any(x is None for x in lst):
            return None
        return torch.stack(lst).to(dtype)

    @staticmethod
    def safe_pad(lst, batch_first=True, dtype=torch.float32):
        """
        Pads tensor list.
        Returns None if any element in the list is None.
        """
        if any(x is None for x in lst):
            return None

        padded = nn.utils.rnn.pad_sequence(lst, batch_first=batch_first).to(dtype)
        return padded
