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
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config
from xanesnet.serialization.splits import load_split_indices
from xanesnet.utils.exceptions import ConfigError

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
        split_ratios: list[float] | None,
        split_indexfile: str | None,
    ) -> None:
        self.dataset_type = dataset_type
        self.datasource = datasource
        self.root = root
        self.preload = preload

        # preloaded dataset
        self.inmemory_dataset: list[Any] = []

        # subsets for splits
        self._subsets: list[Subset] = self._setup_splits(split_ratios, split_indexfile)

    @abstractmethod
    def prepare(self) -> bool:
        """
        Process the raw data and prepare it for use in the model.
        """
        # Check if processing already done
        if all(os.path.exists(f) for f in self.processed_files):
            logging.info("Processed data files already exist. Skipping processing.")
            return True
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            logging.info(f"Created root data directory at: {self.root}")
            return False
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
                preload_data.append(self._load_item(file))
            self.inmemory_dataset = preload_data
        return self.preload

    def _setup_splits(self, split_ratios: list[float] | None, split_indexfile: str | None) -> list[Subset]:
        """
        Create multiple subsets based on the split_indexfile or by generating them from split_ratios.
        """
        subsets: list[Subset] = []

        if split_indexfile is not None:
            logging.info(f"Setting up dataset splits from index file: {split_indexfile}")
            indices_list: list[list[int]] = load_split_indices(split_indexfile)

            for indices in indices_list:
                subsets.append(Subset(self, indices))
        elif split_ratios is not None:
            ratio_sum = sum(split_ratios)
            if not np.isclose(ratio_sum, 1.0):
                raise ConfigError(f"split_ratios must sum to 1.0, but got {ratio_sum}")

            logging.info(f"Setting up dataset splits with ratios: {split_ratios}")

            dataset_size = len(self)
            indices = np.random.permutation(dataset_size).tolist()

            for i, ratio in enumerate(split_ratios):
                if i == len(split_ratios) - 1:
                    end_idx = dataset_size
                else:
                    end_idx = int(dataset_size * ratio)
                split_indices = indices[:end_idx]
                indices = indices[end_idx:]
                subsets.append(Subset(self, split_indices))
        else:
            logging.warning("No split_ratios or split_indexfile provided. Dataset will be created without splits.")
            return []

        return subsets

    def get_subset_indices(self, index: int) -> list[int] | None:
        """
        Return the indices for a specific subset.
        """
        subset = self.get_subset(index)
        if subset is not None:
            return list(subset.indices)
        return None

    def get_all_subset_indices(self) -> list[list[int]]:
        """
        Return the indices for all subsets.
        """
        return [self.get_subset_indices(i) or [] for i in range(len(self._subsets))]

    def get_subset(self, index: int) -> Subset | None:
        """
        Return the subset at the given index.
        """
        if 0 <= index < len(self._subsets):
            return self._subsets[index]
        return None

    @property
    def subsets(self) -> list[Subset]:
        """
        Return all subsets.
        """
        return self._subsets

    @property
    def train_subset(self) -> Subset | None:
        """
        Return the training subset.
        """
        return self.get_subset(0)

    @property
    def valid_subset(self) -> Subset | None:
        """
        Return the validation subset.
        """
        return self.get_subset(1)

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

    @abstractmethod
    def _load_item(self, path: str) -> Any:
        """
        Load a single item from the given path.
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
            return self._load_item(self.processed_files[idx])

    @property
    @abstractmethod
    def signature(self) -> Config:
        """
        Return dataset signature as a dictionary.
        """
        signature = Config(
            {
                "dataset_type": self.dataset_type,
            }
        )
        return signature

    @property
    def processed_dir(self) -> str:
        """
        Path to the processed data directory.
        """
        return self.root

    @property
    def processed_files(self) -> list[str]:
        """
        List of processed data files according to datasource length.
        """
        return [os.path.join(self.processed_dir, f"{i}.pth") for i in range(len(self.datasource))]
