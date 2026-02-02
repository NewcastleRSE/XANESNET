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
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from tqdm import tqdm

from xanesnet.datasources import DataSource

###############################################################################
#################################### CLASS ####################################
###############################################################################


class Dataset(TorchDataset, ABC):
    """
    Abstract base class for datasets.
    """

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
    ) -> None:
        self.dataset_type = dataset_type
        self.datasource = datasource
        self.root = root
        self.preload = preload

        # preloaded dataset
        self.inmemory_dataset: list[Any] = []

        # train/valid split
        self._train_subset: Subset | None = None
        self._valid_subset: Subset | None = None

    @abstractmethod
    def prepare(self) -> bool:
        """
        Process the raw data and prepare it for use in the model.
        """
        # Check if processing already done
        if all(os.path.exists(f) for f in self.processed_files):
            logging.info("Processed data files already exist. Skipping processing.")
            return True
        if os.listdir(self.processed_dir):
            logging.warning("Processed data directory is not empty! Make sure this is intended.")

        return False

    def check_preload(self) -> bool:
        """
        Preload the entire dataset into memory.
        """
        if self.preload:
            logging.info(f"Preloading entire dataset into memory. (# Samples: {len(self)})")
            preload_data = []
            for file in tqdm(self.processed_files, desc="Preloading dataset", total=len(self)):
                preload_data.append(torch.load(file))
            self.inmemory_dataset = preload_data
        return self.preload

    def setup_train_val_split(self, train_ratio: float) -> None:
        """
        Create train and validation subsets based on the specified ratio.
        """
        if not 0.0 <= train_ratio <= 1.0:
            raise ValueError(f"train_ratio must be between 0.0 and 1.0, got {train_ratio}")

        dataset_size = len(self)
        train_size = int(dataset_size * train_ratio)
        valid_size = dataset_size - train_size

        assert train_size + valid_size == dataset_size

        # Generate random indices
        indices = np.random.permutation(dataset_size)
        train_indices = indices[:train_size].tolist()
        valid_indices = indices[train_size:].tolist() if valid_size > 0 else []

        # Create Subset objects
        self._train_subset = Subset(self, train_indices)
        self._valid_subset = Subset(self, valid_indices) if valid_size > 0 else None

        logging.info(
            f"Train/Valid split created: "
            f"train_size={len(self._train_subset)}, "
            f"valid_size={len(self._valid_subset) if self._valid_subset else 0}"
        )

    @property
    def train_subset(self) -> Subset | None:
        """
        Return the training subset. Returns None if setup_train_val_split() hasn't been called.
        """
        return self._train_subset

    @property
    def valid_subset(self) -> Subset | None:
        """
        Return the validation subset. Returns None if no validation split or setup_train_val_split() hasn't been called.
        """
        return self._valid_subset

    @abstractmethod
    def get_dataloader(self) -> Any:
        """
        Returns the dataloader class that should be used.
        """
        ...

    @abstractmethod
    def collate_fn(self, batch: list[Any]) -> Any:
        """
        Collate function for the dataloader.
        """
        ...

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.datasource)

    def __getitem__(self, idx: int) -> Any:
        """
        Return the sample at the given index.
        """
        if self.preload:
            return self.inmemory_dataset[idx]
        else:
            return torch.load(self.processed_files[idx])

    @property
    @abstractmethod
    def signature(self) -> dict[str, Any]:
        """
        Return dataset signature as a dictionary.
        """
        signature = {
            "dataset_type": self.dataset_type,
        }
        return signature

    @property
    def processed_dir(self) -> str:
        """
        Path to the processed data directory.
        """
        return os.path.join(self.root, "processed")

    @property
    def processed_files(self) -> list[str]:
        """
        List of processed data files according to datasource length.
        """
        return [os.path.join(self.processed_dir, f"{i}.pt") for i in range(len(self.datasource))]
