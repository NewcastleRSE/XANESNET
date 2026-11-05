# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Abstract dataset infrastructure shared by all dataset implementations."""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import copy
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.serialization._types import SplitPlan, SplitPlans
from xanesnet.serialization.config import Config
from xanesnet.serialization.splits import load_splits
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.prompts import confirm_yes_no

SavePathFn = Callable[[int], str]


class Dataset(TorchDataset, ABC):
    """Abstract base class for prepared on-disk datasets.

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw data source that yields structures, spectra, and metadata.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to load processed samples into memory after preparation.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional train and validation split ratios.
        split_indexfile: Optional path to serialized split indices.
    """

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
        """Initialize shared dataset state."""
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
        self._subsets: list[list[Subset]] = []

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
            if confirm_yes_no("INPUT NEEDED \t- Empty processed data directory?", default_yes=True):
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
                """Return the canonical path for one per-item output.

                Args:
                    seq: Output sequence number for the current datasource item.
                    offset: First processed sample index assigned to this datasource item.

                Returns:
                    Canonical path for the processed sample.
                """
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

    def setup_splits(self) -> SplitPlans:
        """Create configured subsets and return their split-plan indices.

        Returns:
            Split plans created from the configured ratios or index file.
        """
        if self._split_indexfile is not None:
            logging.info(f"Setting up dataset splits from index file: {self._split_indexfile}")
            split_plans = load_splits(self._split_indexfile)
            self.set_splits(split_plans)
            return split_plans
        elif self._split_ratios is not None:
            ratio_sum = sum(self._split_ratios)
            if not np.isclose(ratio_sum, 1.0):
                raise ConfigError(f"split_ratios must sum to 1.0, but got {ratio_sum}")

            logging.info(f"Setting up dataset splits with ratios: {self._split_ratios}")

            dataset_size = len(self)
            indices = np.random.permutation(dataset_size).tolist()

            split_plan: SplitPlan = []
            plan_subsets: list[Subset] = []
            for i, ratio in enumerate(self._split_ratios):
                if i == len(self._split_ratios) - 1:
                    end_idx = dataset_size
                else:
                    end_idx = int(dataset_size * ratio)
                split_indices = indices[:end_idx]
                indices = indices[end_idx:]
                split_plan.append(split_indices)
                plan_subsets.append(Subset(self, split_indices))
            self._subsets = [plan_subsets]
            return [split_plan]
        else:
            logging.warning("No split_ratios or split_indexfile provided. Dataset will be created without splits.")
            self._subsets = []
            return []

    def set_splits(self, split_plans: SplitPlans) -> None:
        """Set subsets based on the provided split plans.

        Args:
            split_plans: Ordered sample indices for each subset in every plan.
        """
        subsets: list[list[Subset]] = []
        for split_plan in split_plans:
            plan_subsets: list[Subset] = []
            for indices in split_plan:
                plan_subsets.append(Subset(self, indices))
            subsets.append(plan_subsets)
        self._subsets = subsets

    def get_dataset(self, idx: int) -> "Dataset":
        """Return a dataset view using one stored split plan.

        Args:
            idx: Zero-based index of the split plan.

        Returns:
            A shallow dataset copy with subsets for the selected plan.
        """
        if not self._subsets:
            raise ConfigError("Cannot create a dataset view before split plans are set.")

        dataset_view = copy(self)
        dataset_view._subsets = [self._subsets[idx]]
        return dataset_view

    @property
    def has_split_indexfile(self) -> bool:
        """Check whether a split index file was configured.

        Returns:
            Whether this dataset has a configured split index file.
        """
        return self._split_indexfile is not None

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
        subsets = self.subsets
        return [list(subset.indices) for subset in subsets]

    def get_subset(self, index: int) -> Subset | None:
        """Return one configured subset.

        Args:
            index: Subset position in ``self.subsets``.

        Returns:
            The requested subset, or ``None`` when ``index`` is out of range.
        """
        subsets = self.subsets
        if 0 <= index < len(subsets):
            return subsets[index]
        return None

    @property
    def subsets(self) -> list[Subset]:
        """Configured dataset subsets.

        Returns:
            List of subsets created by ``setup_splits``.
        """
        return self._subsets[0] if self._subsets else []

    @property
    def train_subset(self) -> Subset | None:
        """Training subset, when configured.

        Returns:
            First configured subset, or ``None`` when no training subset exists.
        """
        return self.get_subset(0)

    @property
    def valid_subset(self) -> Subset | None:
        """Validation subset, when configured.

        Returns:
            Second configured subset, or ``None`` when no validation subset exists.
        """
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
        """Path to the processed data directory.

        Returns:
            Root directory where processed ``.pth`` files are stored.
        """
        return self.root

    @property
    def processed_files(self) -> list[str]:
        """Processed file paths in canonical sample order.

        Returns:
            File paths ordered by processed sample index.
        """
        return [os.path.join(self.processed_dir, f"{i}.pth") for i in range(len(self))]
