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

"""Concrete ``BasicInferencer`` implementation for XANESNET."""

import time

import torch
from tqdm import tqdm

from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model
from xanesnet.serialization.prediction_writers import PredictionBatch, PredictionWriter

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
        encoding: Composed spectra encoding applied to decode model predictions
            back into the original spectrum space.
        batch_size: Number of samples per inference batch.
        shuffle: Whether to shuffle the data (typically ``False`` for inference).
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of data-loader worker processes.
        inferencer_type: Identifier string for this inferencer type.
        buffer_size: Number of target-site rows buffered before prediction data is
            flushed to disk.
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
        # inferencer params:
        inferencer_type: str,
        buffer_size: int,
    ) -> None:
        """Initialize ``BasicInferencer``."""
        super().__init__(
            dataset,
            model,
            device,
            encoding,
            batch_size,
            shuffle,
            drop_last,
            num_workers,
            inferencer_type,
            buffer_size,
        )

    def _infer_one_epoch(self, writer: PredictionWriter | None) -> None:
        """Run one inference epoch over the data loader.

        Args:
            writer: Prediction writer to accumulate results, or ``None`` to
                discard predictions.
        """
        self.model.eval()

        for batch in tqdm(self.dataloader, desc="Inference"):
            batch.to(self.device)

            inputs = self.batch_processor.input_preparation(batch)
            elements = self.batch_processor.element_preparation(batch)
            inputs = self.batch_processor.encode_input(inputs, elements)

            # Time the forward pass start
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()  # Needed to block CPU until all GPU ops are done
            start_time = time.perf_counter()

            # Forward pass
            with torch.no_grad():
                predictions = self.model(**inputs)

            # Time the forward pass end
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()  # Needed to block CPU until all GPU ops are done
            end_time = time.perf_counter()

            predictions = self.batch_processor.prediction_preparation(batch, predictions)

            predictions = self.batch_processor.decode_target(predictions, elements)

            # Timing fields are broadcast to ``[n_target_sites]`` so they
            # follow the writer's per-target-site leading-dimension contract:
            #   * ``time_per_spectrum`` -- forward-pass duration divided by the
            #     number of spectra produced by this batch.
            # The value is in seconds and covers only the timed model forward
            # pass, excluding data loading and output writing.
            # ``predictions`` after ``prediction_preparation`` already contains
            # exactly the target sites with ground truth (selected via
            # ``target_site_mask`` for masking models, all rows for per-target-site
            # datasets).
            n_target_sites = predictions.shape[0]
            wall_time = end_time - start_time
            time_per_spectrum = torch.full(
                (n_target_sites,),
                wall_time / n_target_sites if n_target_sites > 0 else 0.0,
                dtype=torch.float32,
                device=self.device,
            )
            # Target
            targets = self.batch_processor.target_preparation(batch)

            # Writer add
            if writer is not None:
                prediction_batch: PredictionBatch = {
                    "prediction": predictions,
                    "target": targets,
                    "sample_id": self.batch_processor.sample_id_preparation(batch),
                    "time_per_spectrum": time_per_spectrum,
                }
                target_site_indices = self.batch_processor.target_site_index_preparation(batch)
                if target_site_indices is not None:
                    prediction_batch["target_site_index"] = target_site_indices
                writer.add(prediction_batch)
