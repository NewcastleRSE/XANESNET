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
from typing import Any

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model

from ..base import Runner


class Inferencer(Runner):
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
        loss: dict[str, Any],
        regularizer: dict[str, Any],
        # inferencer params:
        inferencer_type: str,
    ) -> None:
        super().__init__(dataset, model, device, batch_size, shuffle, drop_last, num_workers, loss, regularizer)

        self.inferencer_type = inferencer_type

        # Setup
        self.batch_processor = self._setup_batchprocessor()
        self.dataloader = self._setup_dataloader()
        self.loss = self._setup_loss()
        self.regularizer = self._setup_regularizer()

    def infer(self) -> float:
        """
        Core inference (1 epoch).
        """
        self.model.to(self.device)

        logging.info("Start inference.")
        test_loss = None

        # Run inference
        test_loss, test_regularization, test_total = self._infer_one_epoch()

        logging.info("Finished inference.")

        # Logging
        self._log_epoch_loss(
            test_loss,
            test_regularization,
            test_total,
            None,
            None,
            None,
            None,
        )

        score = test_total

        return score

    def _infer_one_epoch(self) -> tuple[float, float, float]:
        """
        Runs a single inference epoch.
        """
        self.model.eval()

        epoch_loss = 0.0
        epoch_regularization = 0.0
        epoch_total = 0.0

        for batch in self.dataloader:
            batch.to(self.device)

            # Forward pass
            inputs = self.batch_processor.input_preparation(batch)
            predictions = self.model(inputs)

            # Target
            targets = self.batch_processor.target_preparation(batch)

            # Criterion
            loss = self.loss(predictions, targets)

            # Regularization (Should be 'none' most of the time.)
            regularization = self.regularizer(self.model)

            total = loss + regularization

            epoch_loss += loss.item()
            epoch_regularization += regularization.item()
            epoch_total += total.item()

        epoch_loss = epoch_loss / len(self.dataloader)
        epoch_regularization = epoch_regularization / len(self.dataloader)
        epoch_total = epoch_total / len(self.dataloader)

        return epoch_loss, epoch_regularization, epoch_total
