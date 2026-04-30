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

"""Deep-ensemble inferencer implementation for XANESNET."""

import logging
import time
from pathlib import Path

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.serialization.prediction_writers import HDF5Writer, PredictionWriter

from .base import Inferencer
from .registry import InferencerRegistry


@InferencerRegistry.register("ensemble")
class EnsembleInferencer(Inferencer):
    """Inferencer for multiple independently trained models.

    Runs a single data loader and evaluates every ensemble member on each
    batch. Member predictions are post-processed with the configured batch
    processor, stacked along a model dimension, and reduced to a mean
    prediction plus an energy/channel-wise standard deviation. Per-member
    spectra are not persisted.

    ``model_device_policy`` controls how member models are placed on the
    inference device. ``"all"`` keeps every model on the device for faster
    inference when the ensemble fits in memory. ``"sequential"`` moves one
    model at a time to the device, copies its prepared predictions back to CPU,
    and then moves the model back to CPU before evaluating the next member.

    Args:
        dataset: Dataset to run inference on.
        models: Non-empty list of models to evaluate.
        device: Device identifier or :class:`torch.device` instance.
        batch_size: Number of samples per inference batch.
        shuffle: Whether to shuffle the data.
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of data-loader worker processes.
        inferencer_type: Identifier string for this inferencer type.
        model_device_policy: Device-placement policy for ensemble members.
            Supported values are ``"all"`` and ``"sequential"``.
    """

    def __init__(
        self,
        dataset: Dataset,
        models: list[Model],
        device: str | torch.device,
        # runner params:
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        # inferencer params:
        inferencer_type: str,
        model_device_policy: str,
    ) -> None:
        """Initialize ``EnsembleInferencer``."""
        super().__init__(
            dataset,
            models[0],
            device,
            batch_size,
            shuffle,
            drop_last,
            num_workers,
            inferencer_type,
        )
        self.models = models
        self.model_device_policy = model_device_policy

    def infer(self, predictions_save_path: str | Path | None = None) -> None:
        """Run one full ensemble inference pass over the dataset.

        Uses the configured ``model_device_policy`` to place ensemble members
        on the inference device, and moves all members back to CPU before
        returning.

        Args:
            predictions_save_path: Path to write aggregate predictions to
                (HDF5 format). Pass ``None`` to run inference without saving
                results.
        """
        writer = None

        if self.model_device_policy == "all":
            for model in self.models:
                model.to(self.device)
        else:
            for model in self.models:
                model.to(torch.device("cpu"))

        writer = HDF5Writer(predictions_save_path, buffer_size=3) if predictions_save_path is not None else None

        logging.info("Start ensemble inference.")

        try:
            self._infer_one_epoch(writer)
        finally:
            if writer is not None:
                writer.close()
            for model in self.models:
                model.to(torch.device("cpu"))

        logging.info("Finished ensemble inference.")

    def _infer_one_epoch(self, writer: PredictionWriter | None) -> None:
        """Run one ensemble inference epoch over the data loader.

        Args:
            writer: Prediction writer to accumulate aggregate results, or
                ``None`` to discard predictions.

        Raises:
            ValueError: If ensemble member predictions do not share the same
                post-processed shape.
        """
        for model in self.models:
            model.eval()

        for batch in self.dataloader:
            batch.to(self.device)
            inputs = self.batch_processor.input_preparation(batch)

            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            member_predictions: list[torch.Tensor] = []
            with torch.no_grad():
                for model in self.models:
                    if self.model_device_policy == "sequential":
                        model.to(self.device)

                    predictions = model(**inputs)
                    predictions = self.batch_processor.prediction_preparation(batch, predictions)
                    if self.model_device_policy == "sequential":
                        member_predictions.append(predictions.detach().cpu())
                        del predictions
                        model.to(torch.device("cpu"))
                    else:
                        member_predictions.append(predictions)

            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            try:
                stacked_predictions = torch.stack(member_predictions, dim=0)
            except RuntimeError as exc:
                raise ValueError("Ensemble member predictions must have identical shapes after preparation.") from exc

            predictions_mean = stacked_predictions.mean(dim=0)
            predictions_std = stacked_predictions.std(dim=0, unbiased=False)

            n_absorbers = predictions_mean.shape[0]
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

            targets = self.batch_processor.target_preparation(batch)

            if writer is not None:
                writer.add(
                    {
                        "prediction": predictions_mean,
                        "prediction_std": predictions_std,
                        "target": targets,
                        "file_name": self.batch_processor.file_name_extraction(batch),
                        "forward_time": forward_time,
                        "forward_time_pass": forward_time_pass,
                    }
                )
