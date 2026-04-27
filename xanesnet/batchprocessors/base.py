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

from abc import ABC, abstractmethod

# TODO do we need this ? Why do we need this? Can we get rid of it?
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from xanesnet.datasets import Dataset


class BatchProcessor(ABC):
    """
    Base class for batch processors.
    Converts a dataset batch into the correct inputs and targets for a specific model.
    """

    @abstractmethod
    def input_preparation(self, batch: Any) -> dict[str, Any]:
        """
        Prepares the model inputs from a batch.
        """
        ...

    def input_preparation_single(self, dataset: "Dataset", index: int) -> dict[str, Any]:
        """
        Prepares the model inputs from a single sample.
        Collates the sample into a batch of size 1 and delegates to input_preparation.
        """
        sample = dataset[index]
        batch = dataset.collate_fn([sample])
        return self.input_preparation(batch)

    def prediction_preparation(self, batch: Any, predictions: torch.Tensor) -> torch.Tensor:
        """
        Prepares the model predictions for loss/regularization computation.
        By default, returns the raw predictions.
        """
        return predictions

    @abstractmethod
    def target_preparation(self, batch: Any) -> torch.Tensor:
        """
        Prepares the model targets from a batch.
        """
        ...

    def target_preparation_single(self, dataset: "Dataset", index: int) -> torch.Tensor:
        """
        Prepares the model targets from a single sample.
        Collates the sample into a batch of size 1 and delegates to target_preparation.
        """
        sample = dataset[index]
        batch = dataset.collate_fn([sample])
        return self.target_preparation(batch)

    @abstractmethod
    def file_name_extraction(self, batch: Any) -> np.ndarray:
        """
        Extracts file names from a batch.
        """
        ...
