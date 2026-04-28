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

"""Concrete ``BasicInferencer`` implementation for XANESNET."""

import time

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.serialization.prediction_writers import PredictionWriter

from .base import Inferencer
from .registry import InferencerRegistry


@InferencerRegistry.register("basic")
class BasicInferencer(Inferencer):
    """Basic single-process inferencer.

    Runs model inference over a dataset and optionally writes predictions to
    disk through the configured prediction writer.

    Args:
        dataset: Dataset to run inference on.
        model: Model to evaluate.
        device: Device identifier or :class:`torch.device` instance.
        batch_size: Number of samples per inference batch.
        shuffle: Whether to shuffle the data (typically ``False`` for inference).
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of data-loader worker processes.
        inferencer_type: Identifier string for this inferencer type.
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
        """Initialize ``BasicInferencer``."""
        super().__init__(
            dataset,
            model,
            device,
            batch_size,
            shuffle,
            drop_last,
            num_workers,
            inferencer_type,
        )

    def _infer_one_epoch(self, writer: PredictionWriter | None) -> None:
        """Run one inference epoch over the data loader.

        Args:
            writer: Prediction writer to accumulate results, or ``None`` to
                discard predictions.
        """
        self.model.eval()

        # TODO some progress bar?

        for batch in self.dataloader:
            batch.to(self.device)

            # Prepare inputs
            inputs = self.batch_processor.input_preparation(batch)

            # Time the forward pass start
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()  # Needed to block CPU until all GPU ops are done
            start_time = time.perf_counter()

            # Forward pass
            predictions = self.model(**inputs)

            # Time the forward pass end
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()  # Needed to block CPU until all GPU ops are done
            end_time = time.perf_counter()

            predictions = self.batch_processor.prediction_preparation(batch, predictions)

            # Two timing fields, both broadcast to ``[n_absorbers]`` so they
            # follow the writer's per-absorber leading-dim contract:
            #   * ``forward_time``      -- amortized per-absorber cost: the
            #     wall-clock duration of this forward pass divided by the
            #     number of absorbers with ground truth produced by it.
            #     Useful as the time budget attributable to a single spectrum.
            #   * ``forward_time_pass`` -- raw wall-clock duration of the
            #     forward pass, repeated for every absorber it produced.
            #     Independent of batch size / multi-absorber count.
            # ``predictions`` after ``prediction_preparation`` already contains
            # exactly the absorbers with ground truth (selected via
            # ``absorber_mask`` for masking models, all rows for per-absorber
            # datasets).
            n_absorbers = predictions.shape[0]
            wall_time = end_time - start_time
            forward_time = torch.full(
                (n_absorbers,),
                wall_time / n_absorbers if n_absorbers > 0 else 0.0,
                dtype=torch.float32,
                device=self.device,
            )
            forward_time_pass = torch.full(
                (n_absorbers,),
                wall_time,
                dtype=torch.float32,
                device=self.device,
            )

            # Target
            targets = self.batch_processor.target_preparation(batch)

            # Writer add
            if writer is not None:
                writer.add(
                    {
                        # Required:
                        "prediction": predictions,
                        "target": targets,
                        # Optional:
                        "file_name": self.batch_processor.file_name_extraction(batch),
                        "forward_time": forward_time,
                        "forward_time_pass": forward_time_pass,
                    }
                )
