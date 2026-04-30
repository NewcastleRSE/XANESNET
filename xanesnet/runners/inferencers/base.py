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

"""Abstract base class for all XANESNET inferencers."""

import logging
from abc import abstractmethod
from pathlib import Path

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.serialization.prediction_writers import HDF5Writer, PredictionWriter

from ..base import Runner


class Inferencer(Runner):
    """Abstract base class for all XANESNET inferencers.

    Args:
        dataset: Dataset to run inference on.
        model: Model to evaluate.
        device: Device identifier or :class:`torch.device` instance.
        batch_size: Number of samples per inference batch.
        shuffle: Whether to shuffle the data (typically ``False`` for inference).
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of data-loader worker processes.
        inferencer_type: Identifier string for the concrete inferencer type.
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
        # inferencer params:
        inferencer_type: str,
    ) -> None:
        """Initialize ``Inferencer``."""
        super().__init__(dataset, model, device, batch_size, shuffle, drop_last, num_workers)

        self.inferencer_type = inferencer_type

        # Setup
        self.batch_processor = self._setup_batchprocessor()
        self.dataloader = self._setup_dataloader()

    def infer(self, predictions_save_path: str | Path | None = None) -> None:
        """Run one full inference pass over the dataset.

        Args:
            predictions_save_path: Path to write predictions to (HDF5 format).
                Pass ``None`` to run inference without saving results.
        """
        self.model.to(self.device)

        # You can change the writer to another implementation if needed (e.g., NumpyWriter)
        # TODO adjust buffer size? Should we have a config? Or set to a reasonable default?
        writer = HDF5Writer(predictions_save_path, buffer_size=3) if predictions_save_path is not None else None

        logging.info("Start inference.")

        try:
            # Run inference
            self._infer_one_epoch(writer)
        finally:
            if writer is not None:
                writer.close()
            self.model.to(torch.device("cpu"))

        logging.info("Finished inference.")

    @abstractmethod
    def _infer_one_epoch(self, writer: PredictionWriter | None) -> None:
        """Run one inference epoch over the data loader.

        Args:
            writer: Prediction writer to accumulate results, or ``None`` to
                discard predictions.
        """
        ...
