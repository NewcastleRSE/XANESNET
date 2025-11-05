"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

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

from torch.utils.data import Dataset
from collections.abc import Sequence
from pathlib import Path
from typing import Union, List, Any, Callable

from torch import Tensor
from torch_geometric.io import fs

from xanesnet.utils.mode import Mode

IndexType = Union[slice, Tensor, np.ndarray, Sequence]

###############################################################################
################################## CLASSES ####################################
###############################################################################


class BaseDataset(Dataset):
    """An abstract base class for xanesnet datasets."""

    def __init__(
        self,
        root: Path,
        xyz_path: List[Path] | Path = None,
        xanes_path: List[Path] | Path = None,
        mode: Mode = None,
        descriptors: List = None,
        **kwargs,
    ):
        super().__init__()

        self.root = root
        self.xyz_path = xyz_path
        self.xanes_path = xanes_path
        self.mode = mode
        self.descriptors = descriptors
        self.preload = kwargs.get("preload", True)

        self.config = {}
        self.preload_dataset = []
        self.file_names = None

        self.set_file_names()
        self._process()

    def set_file_names(self):
        """Set a list of file names (stems) used in the dataset."""
        raise NotImplementedError

    def process(self):
        """Processes the dataset and save to the self.processed_dir folder."""
        raise NotImplementedError

    def collate_fn(self, batch):
        """Custom collate function to handle a list of Data objects."""
        raise NotImplementedError

    @property
    def x_size(self) -> Union[int, List[int]]:
        """Size of the feature array."""
        raise NotImplementedError

    @property
    def y_size(self) -> Union[int, List[int]]:
        """Size of the label array."""
        raise NotImplementedError

    @property
    def indices(self) -> Sequence:
        """A list of integer indices corresponding to the data points."""
        return list(range(len(self.file_names)))

    @property
    def processed_file_names(self) -> List[str]:
        """A list of all processed file names."""
        return [f"{stem}.pt" for i, stem in enumerate(self.file_names)]

    @property
    def processed_dir(self) -> str:
        """The directory containing processed datasets"""
        return os.path.join(self.root, "processed")

    @property
    def processed_paths(self) -> List[str]:
        """A list of absolute paths to all processed files."""
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [os.path.join(self.processed_dir, f) for f in to_list(files)]

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: Union[int, np.integer, IndexType]):
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
        Checks if processing is complete and runs the process() if not.
        """
        if files_exist(self.processed_paths):
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

    def unique_path(self, path) -> Path:
        if isinstance(path, list):
            if len(path) > 1:
                raise ValueError(
                    "Dataset does not support multiple paths. Please provide only one."
                )
            path = path[0] if path else None

        return Path(path) if path is not None else None

    def list_path(self, path) -> List[Path]:
        if isinstance(path, list):
            return [Path(x) for x in path] if path else None
        return [path] if path is not None else None

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
        config = kwargs.copy()

        # Extract parameters from the local_vars, excluding 'self' and '__class__'
        args_dict = {key: val for key, val in args.items() if key in ["type", "params"]}

        # config.update(params)
        config.update(args_dict)

        self.config = config


def files_exist(files: List[str]) -> bool:
    return len(files) != 0 and all([fs.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
