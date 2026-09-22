# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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
#   ...

"""Abstract base class for all XANESNET runners."""

import logging
from abc import ABC
from typing import Any

import torch

from xanesnet.batchprocessors import BatchProcessor, BatchProcessorRegistry
from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model


class Runner(ABC):
    """Abstract base class for all XANESNET runners (trainers and inferencers).

    Args:
        dataset: Dataset to iterate over.
        model: Model to run.
        device: Device identifier (e.g. ``'cpu'``, ``'cuda'``) or :class:`torch.device` instance.
        encoding: Composed spectra encoding used to encode targets before the
            loss and decode predictions afterwards.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at each epoch.
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of worker processes for data loading.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        device: str | torch.device,
        encoding: SpectraEncoding,
        # runner params:
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
    ) -> None:
        """Initialize ``Runner``."""
        self.dataset = dataset
        self.model = model
        self.device = device
        self.encoding = encoding

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    def _setup_batchprocessor(self) -> BatchProcessor:
        """Instantiate the batch processor appropriate for the current
        dataset/model pair.

        The stored encoding is forwarded so the batch processor can apply
        it internally in its preparation methods.

        Returns:
            A configured :class:`BatchProcessor` instance.
        """
        batchprocessor = BatchProcessorRegistry.create(
            (self.dataset.dataset_type, self.model.model_type),
            encoding=self.encoding,
        )
        return batchprocessor

    def _build_dataloader(self, data: Any, shuffle: bool, drop_last: bool) -> Any:
        """Build a data loader with the runner's common loading settings.

        Args:
            data: Dataset or subset to iterate over.
            shuffle: Whether to shuffle samples before batching.
            drop_last: Whether to drop the final incomplete batch.

        Returns:
            A configured data-loader instance.
        """
        dataloader_cls = self.dataset.get_dataloader()

        dataloader = dataloader_cls(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_fn,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=torch.device(self.device).type == "cuda",
            persistent_workers=False if self.num_workers == 0 else True,
            prefetch_factor=None if self.num_workers == 0 else 2,
        )

        return dataloader

    def _setup_dataloader(self) -> Any:
        """Build a data loader over the full dataset.

        Returns:
            A configured data-loader instance.
        """
        return self._build_dataloader(self.dataset, shuffle=self.shuffle, drop_last=self.drop_last)

    @staticmethod
    def _log_epoch_loss(
        loss: float,
        regularization: float,
        total: float,
        epoch: int,
        valid_loss: float | None = None,
        valid_regularization: float | None = None,
        valid_total: float | None = None,
    ) -> None:
        """Log training (and optionally validation) metrics for one epoch.

        Args:
            loss: Mean training loss for the epoch.
            regularization: Mean training regularization term for the epoch.
            total: Mean total training loss (``loss + regularization``) for the epoch.
            valid_loss: Mean validation loss, or ``None`` if not computed.
            valid_regularization: Mean validation regularization term, or ``None``.
            valid_total: Mean total validation loss, or ``None``.
            epoch: Current epoch index.
        """
        epoch_str = f"Epoch {epoch:03d} | "
        train_str = f"Loss: {loss:.6f} | Reg: {regularization:.6f} | Total: {total:.6f}"

        if valid_total is not None:
            valid_str = (
                f"Valid Loss: {valid_loss:.6f} | Valid Reg: {valid_regularization:.6f} | Valid Total: {valid_total:.6f}"
            )
            logging.info(f"{epoch_str}{train_str} | {valid_str}")
        else:
            logging.info(f"{epoch_str}{train_str}")
