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

"""Abstract base class for all XANESNET runners."""

import logging
from abc import ABC
from typing import Any

import torch

from xanesnet.batchprocessors import BatchProcessor, BatchProcessorRegistry
from xanesnet.datasets import Dataset
from xanesnet.models import Model


class Runner(ABC):
    """Abstract base class for all XANESNET runners (trainers and inferencers).

    Args:
        dataset: Dataset to iterate over.
        model: Model to run.
        device: Device identifier (e.g. ``'cpu'``, ``'cuda'``) or :class:`torch.device` instance.
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
        # runner params:
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.device = device

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    def _setup_batchprocessor(self) -> BatchProcessor:
        """Instantiate the batch processor appropriate for the current dataset/model pair.

        Returns:
            A configured :class:`BatchProcessor` instance.
        """
        batchprocessor = BatchProcessorRegistry.get(self.dataset.dataset_type, self.model.model_type)()
        return batchprocessor

    def _setup_dataloader(self) -> Any:
        """Build a data loader over the full dataset.

        Returns:
            A configured data-loader instance.
        """
        dataloader_cls = self.dataset.get_dataloader()

        dataloader = dataloader_cls(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.dataset.collate_fn,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

        return dataloader

    @staticmethod
    def _log_epoch_loss(
        loss: float,
        regularization: float,
        total: float,
        valid_loss: float | None = None,
        valid_regularization: float | None = None,
        valid_total: float | None = None,
        epoch: int | None = None,
    ) -> None:
        """Log training (and optionally validation) metrics for one epoch.

        Args:
            loss: Mean training loss for the epoch.
            regularization: Mean training regularization term for the epoch.
            total: Mean total training loss (``loss + regularization``) for the epoch.
            valid_loss: Mean validation loss, or ``None`` if not computed.
            valid_regularization: Mean validation regularization term, or ``None``.
            valid_total: Mean total validation loss, or ``None``.
            epoch: Current epoch index, or ``None`` to omit the epoch prefix.
        """
        epoch_str = f"Epoch {epoch:03d} | " if epoch is not None else ""
        train_str = f"Loss: {loss:.6f} | Reg: {regularization:.6f} | Total: {total:.6f}"

        if valid_total is not None:
            valid_str = (
                f"Valid Loss: {valid_loss:.6f} | Valid Reg: {valid_regularization:.6f} | Valid Total: {valid_total:.6f}"
            )
            logging.info(f"{epoch_str}{train_str} | {valid_str}")
        else:
            logging.info(f"{epoch_str}{train_str}")
