# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Abstract base class for all XANESNET batch processors."""

from abc import ABC, abstractmethod

# TODO do we need this ? Why do we need this? Can we get rid of it?
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from xanesnet.datasets import Dataset


class BatchProcessor(ABC):
    """Abstract base class for batch processors.

    Converts a dataset batch into the correct model inputs and targets for a specific model
    architecture. Subclasses must implement ``input_preparation``, ``target_preparation``,
    and ``file_name_extraction``.
    """

    @abstractmethod
    def input_preparation(self, batch: Any) -> dict[str, Any]:
        """Prepare model input tensors from a batch.

        Args:
            batch: A collated batch produced by the dataset's ``collate_fn``.

        Returns:
            Dict mapping argument names to tensors (or related objects) expected
            by the model's ``forward`` method.
        """
        ...

    def input_preparation_single(self, dataset: "Dataset", index: int) -> dict[str, Any]:
        """Prepare model inputs from a single dataset sample.

        Collates the sample at ``index`` into a batch of size 1 and delegates
        to :meth:`input_preparation`.

        Args:
            dataset: The dataset to draw the sample from.
            index: Index of the sample within the dataset.

        Returns:
            Dict mapping argument names to tensors expected by the model.
        """
        sample = dataset[index]
        batch = dataset.collate_fn([sample])
        return self.input_preparation(batch)

    def prediction_preparation(self, batch: Any, predictions: torch.Tensor) -> torch.Tensor:
        """Post-process raw model predictions before loss computation.

        The default implementation returns predictions unchanged. Override in subclasses
        to apply masking or other per-batch transformations (e.g. selecting absorber atoms
        from a per-atom output tensor).

        Args:
            batch: The collated batch (may carry masks or indices needed for post-processing).
            predictions: Raw model output tensor.

        Returns:
            Post-processed predictions tensor.
        """
        return predictions

    @abstractmethod
    def target_preparation(self, batch: Any) -> torch.Tensor:
        """Prepare the target tensor from a batch.

        Args:
            batch: A collated batch produced by the dataset's ``collate_fn``.

        Returns:
            Target tensor for loss computation.
        """
        ...

    def target_preparation_single(self, dataset: "Dataset", index: int) -> torch.Tensor:
        """Prepare the target tensor from a single dataset sample.

        Collates the sample at ``index`` into a batch of size 1 and delegates
        to :meth:`target_preparation`.

        Args:
            dataset: The dataset to draw the sample from.
            index: Index of the sample within the dataset.

        Returns:
            Target tensor for loss computation.
        """
        sample = dataset[index]
        batch = dataset.collate_fn([sample])
        return self.target_preparation(batch)

    @abstractmethod
    def file_name_extraction(self, batch: Any) -> np.ndarray:
        """Extract file name identifiers from a batch.

        Args:
            batch: A collated batch produced by the dataset's ``collate_fn``.

        Returns:
            Array of file name strings. ``(batch_size,)``
        """
        ...
