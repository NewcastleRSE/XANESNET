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
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.utils.mode import Mode

###############################################################################
#################################### CLASS ####################################
###############################################################################


class Dataset(ABC):
    """
    Abstract base class for datasets.
    All dataset classes should inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        mode: Mode,
        preload: bool,
        params: dict,
    ):
        self.dataset_type = dataset_type
        self.datasource = datasource
        self.root = root
        self.mode = mode
        self.preload = preload
        self.params = params

        # preloaded dataset
        self.inmemory_dataset = None

    @abstractmethod
    def process(self) -> bool:
        """
        Process the raw data and prepare it for use in the model.
        This method should be implemented by all subclasses.
        """
        # Check if processing already done
        if all(os.path.exists(f) for f in self.processed_files):
            logging.info("Processed data files already exist. Skipping processing.")
            return True
        if os.listdir(self.processed_dir):
            logging.warning("Processed data directory is not empty! Make sure this is intended.")
        return False

    def check_preload(self) -> List[torch.Tensor]:
        """Preload the entire dataset into memory."""
        if self.preload:
            logging.info(f"Preloading entire dataset into memory. (# Samples: {len(self)})")
            preload_data = []
            for file in tqdm(self.processed_files, desc="Preloading dataset", total=len(self)):
                preload_data.append(torch.load(file))
            self.inmemory_dataset = preload_data

    @abstractmethod
    def get_dataloader(self):
        """Returns the dataloader class that should be used."""
        return None

    @property
    def signature(self) -> dict:
        """Return dataset signature as a dictionary."""
        signature = {
            "dataset_type": self.dataset_type,
            "params": self.params,
        }
        return signature

    @property
    def metadata(self) -> dict:
        """Return dataset metadata as a dictionary."""
        metadata = {}
        return metadata

    @property
    def processed_dir(self) -> str:
        """Path to the processed data directory."""
        return os.path.join(self.root, "processed")

    @property
    def processed_files(self) -> List[str]:
        """List of processed data files according to datasource length."""
        return [os.path.join(self.processed_dir, f"{i}.pt") for i in range(len(self.datasource))]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.datasource)

    def __getitem__(self, idx):
        """Return the sample at the given index."""
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            if self.preload:
                return self.inmemory_dataset[idx]
            else:
                return torch.load(self.processed_paths[idx])
        else:
            raise NotImplementedError("Indexing not implemented yet!")
            # TODO : Implement advanced indexing if needed

    @abstractmethod
    def collate_fn(self, batch):
        """
        Collate function for the dataloader.
        """
        pass
