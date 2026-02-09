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
import time
from pathlib import Path

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.serialization.prediction_writers import HDF5Writer, PredictionWriter

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
        # inferencer params:
        inferencer_type: str,
    ) -> None:
        super().__init__(dataset, model, device, batch_size, shuffle, drop_last, num_workers)

        self.inferencer_type = inferencer_type

        # Setup
        self.batch_processor = self._setup_batchprocessor()
        self.dataloader = self._setup_dataloader()

    def infer(self, predictions_save_path: str | Path | None = None) -> None:
        """
        Core inference (1 epoch).
        """
        self.model.to(self.device)

        # You can change the writer to another implementation if needed (e.g., NumpyWriter)
        writer = HDF5Writer(predictions_save_path, buffer_size=3) if predictions_save_path is not None else None

        logging.info("Start inference.")

        # Run inference
        self._infer_one_epoch(writer)

        logging.info("Finished inference.")

        writer.close() if writer is not None else None

    def _infer_one_epoch(self, writer: PredictionWriter | None) -> None:
        """
        Runs a single inference epoch.
        """
        self.model.eval()

        for batch in self.dataloader:
            batch.to(self.device)

            # Prepare inputs
            inputs = self.batch_processor.input_preparation(batch)

            # Time the forward pass start
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()  # Needed to block CPU until all GPU ops are done
            start_time = time.perf_counter()

            # Forward pass
            predictions = self.model(inputs)

            # Time the forward pass end
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()  # Needed to block CPU until all GPU ops are done
            end_time = time.perf_counter()

            # Create per-sample time tensor
            # averaged if batch size > 1
            per_sample_time = (end_time - start_time) / predictions.shape[0]
            forward_time = torch.full(
                (predictions.shape[0],),
                per_sample_time,
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
                        "input": inputs,
                        "sample_id": self.batch_processor.sample_id_extraction(batch),
                        "forward_time": forward_time,
                    }
                )
