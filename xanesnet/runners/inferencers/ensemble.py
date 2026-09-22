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

"""Deep-ensemble inferencer implementation for XANESNET."""

import time

import torch

from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model
from xanesnet.serialization.prediction_writers import PredictionBatch, PredictionWriter

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
        encoding: Composed spectra encoding applied to decode each member's
            predictions before they are reduced to mean and standard deviation
            in the original spectrum space.
        batch_size: Number of samples per inference batch.
        shuffle: Whether to shuffle the data.
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of data-loader worker processes.
        inferencer_type: Identifier string for this inferencer type.
        buffer_size: Number of target-site rows buffered before prediction data is
            flushed to disk.
        model_device_policy: Device-placement policy for ensemble members.
            Supported values are ``"all"`` and ``"sequential"``.
    """

    def __init__(
        self,
        dataset: Dataset,
        models: list[Model],
        device: str | torch.device,
        encoding: SpectraEncoding,
        # runner params:
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        # inferencer params:
        inferencer_type: str,
        buffer_size: int,
        model_device_policy: str,
    ) -> None:
        """Initialize ``EnsembleInferencer``."""
        super().__init__(
            dataset,
            models[0],
            device,
            encoding,
            batch_size,
            shuffle,
            drop_last,
            num_workers,
            inferencer_type,
            buffer_size,
        )
        self.models = models
        self.model_device_policy = model_device_policy

    def _setup_inference_models(self) -> None:
        """Move ensemble members according to ``model_device_policy`` before inference."""
        if self.model_device_policy == "all":
            for model in self.models:
                model.to(self.device)
        else:
            for model in self.models:
                model.to(torch.device("cpu"))

    def _teardown_inference_models(self) -> None:
        """Move all ensemble members back to CPU after inference."""
        for model in self.models:
            model.to(torch.device("cpu"))

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
            elements = self.batch_processor.element_preparation(batch)
            inputs = self.batch_processor.encode_input(inputs, elements)

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
                    predictions = self.batch_processor.decode_target(predictions, elements)
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

            n_target_sites = predictions_mean.shape[0]
            wall_time = end_time - start_time
            time_per_spectrum = torch.full(
                (n_target_sites,),
                wall_time / n_target_sites if n_target_sites > 0 else 0.0,
                dtype=torch.float32,
                device=self.device,
            )
            targets = self.batch_processor.target_preparation(batch)

            if writer is not None:
                prediction_batch: PredictionBatch = {
                    "prediction": predictions_mean,
                    "prediction_std": predictions_std,
                    "target": targets,
                    "sample_id": self.batch_processor.sample_id_preparation(batch),
                    "time_per_spectrum": time_per_spectrum,
                }
                target_site_indices = self.batch_processor.target_site_index_preparation(batch)
                if target_site_indices is not None:
                    prediction_batch["target_site_index"] = target_site_indices
                writer.add(prediction_batch)
