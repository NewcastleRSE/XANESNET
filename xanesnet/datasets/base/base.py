# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either Version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

"""Abstract dataset infrastructure shared by all dataset implementations."""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config
from xanesnet.serialization.splits import load_split_indices
from xanesnet.utils.exceptions import ConfigError

SavePathFn = Callable[[int], str]


class Dataset(TorchDataset, ABC):
    """Abstract base class for prepared on-disk datasets."""

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
        skip_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
    ) -> None:
        """Initialize shared dataset state.

        Args:
            dataset_type: Registered dataset type name.
            datasource: Raw data source that yields structures, spectra, and metadata.
            root: Directory that stores processed ``.pth`` files.
            preload: Whether to load processed samples into memory after preparation.
            skip_prepare: Whether to reuse existing processed files.
            split_ratios: Optional train/validation/test split ratios.
            split_indexfile: Optional path to serialized split indices.
        """
        self.dataset_type = dataset_type
        self.datasource = datasource
        self.root = root
        self.preload = preload
        self.skip_prepare = skip_prepare

        # dataset length
        self._length: int = -1  # will be set in prepare()

        # preloaded dataset
        self.inmemory_dataset: list[Any] = []

        # splits
        self._split_ratios = split_ratios
        self._split_indexfile = split_indexfile
        self._subsets: list[Subset] = []

    def prepare(self) -> bool:
        """Process raw datasource items sequentially.

        Returns:
            ``True`` when preparation or reuse of existing files succeeded.
        """
        return self._prepare_sequential()

    def _prepare_processed_dir(self) -> bool:
        """Prepare the processed data directory.

        Returns:
            ``True`` when existing processed files should be used and no
            processing should run, otherwise ``False``.
        """
        if self.skip_prepare:
            logging.info("skip_prepare is True. Skipping data preparation.")
            self._length = sum(1 for f in os.listdir(self.processed_dir) if f.endswith(".pth"))
            return True

        # Check root directory and processed data
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            logging.info(f"Created processed data directory at: {self.processed_dir}")
            return False
        if os.listdir(self.processed_dir):
            logging.warning("Processed data directory is not empty! Make sure this is intentional.")
            response = input("INPUT NEEDED \t- Empty processed data directory? [Y/n]: ").strip().lower()
            if response in ("y", ""):
                shutil.rmtree(self.processed_dir)
                os.makedirs(self.processed_dir)
                logging.info(f"Cleared processed data directory: {self.processed_dir}")
            else:
                raise ConfigError("Processed data directory is not empty. Please clear it before proceeding.")

        return False

    @abstractmethod
    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one raw datasource item and save all processed samples.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps a per-item sequence number to the
                output path for that processed sample.

        Returns:
            Number of processed files written for ``idx``.
        """
        ...

    def _prepare_sequential(self) -> bool:
        """Process all datasource items sequentially using ``_prepare_single``.

        Returns:
            ``True`` when processing completed or existing files were reused.
        """
        skip_processing = self._prepare_processed_dir()
        if skip_processing:
            return True

        counter = 0
        for idx in tqdm(range(len(self.datasource)), desc="Processing data", total=len(self.datasource)):

            def save_path_fn(seq: int, offset: int = counter) -> str:
                """Return the canonical path for one per-item output."""
                return os.path.join(self.processed_dir, f"{offset + seq}.pth")

            counter += self._prepare_single(idx, save_path_fn)

        self._length = counter
        return True

    def check_preload(self) -> bool:
        """Load processed samples into memory when preloading is enabled.

        Returns:
            Whether the dataset is configured for in-memory preloading.
        """
        if self.preload:
            logging.info(f"Preloading entire dataset into memory. (# Samples: {len(self)})")
            preload_data = []
            for file in tqdm(self.processed_files, desc="Preloading dataset", total=len(self)):
                preload_data.append(self._load_item(file))
            self.inmemory_dataset = preload_data
        return self.preload

    def setup_splits(self) -> None:
        """Create subsets from an index file or configured split ratios."""
        subsets: list[Subset] = []

        if self._split_indexfile is not None:
            logging.info(f"Setting up dataset splits from index file: {self._split_indexfile}")
            indices_list: list[list[int]] = load_split_indices(self._split_indexfile)

            for indices in indices_list:
                subsets.append(Subset(self, indices))
        elif self._split_ratios is not None:
            ratio_sum = sum(self._split_ratios)
            if not np.isclose(ratio_sum, 1.0):
                raise ConfigError(f"split_ratios must sum to 1.0, but got {ratio_sum}")

            logging.info(f"Setting up dataset splits with ratios: {self._split_ratios}")

            dataset_size = len(self)
            indices = np.random.permutation(dataset_size).tolist()

            for i, ratio in enumerate(self._split_ratios):
                if i == len(self._split_ratios) - 1:
                    end_idx = dataset_size
                else:
                    end_idx = int(dataset_size * ratio)
                split_indices = indices[:end_idx]
                indices = indices[end_idx:]
                subsets.append(Subset(self, split_indices))
        else:
            logging.warning("No split_ratios or split_indexfile provided. Dataset will be created without splits.")
            self._subsets = []
            return

        self._subsets = subsets

    def get_subset_indices(self, index: int) -> list[int] | None:
        """Return indices for one configured subset.

        Args:
            index: Subset position in ``self.subsets``.

        Returns:
            Subset indices, or ``None`` when ``index`` is out of range.
        """
        subset = self.get_subset(index)
        if subset is not None:
            return list(subset.indices)
        return None

    def get_all_subset_indices(self) -> list[list[int]]:
        """Return indices for all configured subsets.

        Returns:
            A list of subset index lists.
        """
        return [self.get_subset_indices(i) or [] for i in range(len(self._subsets))]

    def get_subset(self, index: int) -> Subset | None:
        """Return one configured subset.

        Args:
            index: Subset position in ``self.subsets``.

        Returns:
            The requested subset, or ``None`` when ``index`` is out of range.
        """
        if 0 <= index < len(self._subsets):
            return self._subsets[index]
        return None

    @property
    def subsets(self) -> list[Subset]:
        """Configured dataset subsets."""
        return self._subsets

    @property
    def train_subset(self) -> Subset | None:
        """Training subset, when configured."""
        return self.get_subset(0)

    @property
    def valid_subset(self) -> Subset | None:
        """Validation subset, when configured."""
        return self.get_subset(1)

    def get_dataloader(self) -> type[torch.utils.data.DataLoader]:
        """Return the dataloader class for this dataset.

        Returns:
            The dataloader class expected by training and inference code.
        """
        return torch.utils.data.DataLoader

    @abstractmethod
    def collate_fn(self, batch: list[Any]) -> Any:
        """Collate raw items into one dataloader batch.

        Args:
            batch: Samples loaded by ``__getitem__``.

        Returns:
            A framework-specific batch object.
        """
        ...

    @abstractmethod
    def _load_item(self, path: str) -> Any:
        """Load one processed sample from disk.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Loaded sample object.
        """
        ...

    def __len__(self) -> int:
        """Return the number of processed samples.

        Returns:
            Number of processed samples available on disk or in memory.
        """
        if self._length < 0:
            raise ValueError("Dataset length not set. Make sure to call prepare() before using the dataset.")
        return self._length

    def __getitem__(self, idx: int) -> Any:
        """Return one processed sample.

        Args:
            idx: Processed sample index.

        Returns:
            Loaded sample from memory or disk.
        """
        if self.preload:
            return self.inmemory_dataset[idx]
        else:
            return self._load_item(self.processed_files[idx])

    @property
    @abstractmethod
    def signature(self) -> Config:
        """Dataset configuration signature.

        Returns:
            Configuration values that identify this dataset instance.
        """
        signature = Config(
            {
                "dataset_type": self.dataset_type,
            }
        )
        return signature

    @property
    def processed_dir(self) -> str:
        """Path to the processed data directory."""
        return self.root

    @property
    def processed_files(self) -> list[str]:
        """Processed file paths in canonical sample order."""
        return [os.path.join(self.processed_dir, f"{i}.pth") for i in range(len(self))]
