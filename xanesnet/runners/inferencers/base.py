# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Abstract base class for all XANESNET inferencers."""

import logging
from abc import abstractmethod
from pathlib import Path

import torch

from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model
from xanesnet.serialization.prediction_writers import HDF5Writer, PredictionWriter

from ..base import Runner


class Inferencer(Runner):
    """Abstract base class for all XANESNET inferencers.

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
        inferencer_type: Identifier string for the concrete inferencer type.
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
        """Initialize ``Inferencer``."""
        super().__init__(dataset, model, device, encoding, batch_size, shuffle, drop_last, num_workers)

        self.inferencer_type = inferencer_type
        self.buffer_size = buffer_size

        # Setup
        self.batch_processor = self._setup_batchprocessor()
        self.dataloader = self._setup_dataloader()

    def infer(self, predictions_save_path: str | Path | None = None) -> None:
        """Run one full inference pass over the dataset.

        Args:
            predictions_save_path: Path to write predictions to (HDF5 format).
                Pass ``None`` to run inference without saving results.
        """
        writer: PredictionWriter | None = None

        logging.info("Start inference.")

        try:
            self._setup_inference_models()
            writer = self._setup_prediction_writer(predictions_save_path)
            self._infer_one_epoch(writer)
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                self._teardown_inference_models()

        logging.info("Finished inference.")

    def _setup_prediction_writer(self, predictions_save_path: str | Path | None) -> PredictionWriter | None:
        """Instantiate the prediction writer for an inference run.

        Args:
            predictions_save_path: Path to write predictions to (HDF5 format),
                or ``None`` to disable writing.

        Returns:
            A configured prediction writer, or ``None`` when predictions should
            not be saved. The writer uses ``self.buffer_size`` as its flush
            threshold.
        """
        if predictions_save_path is None:
            return None

        return HDF5Writer(predictions_save_path, buffer_size=self.buffer_size)

    def _setup_inference_models(self) -> None:
        """Move models to the inference device before a run starts."""
        self.model.to(self.device)

    def _teardown_inference_models(self) -> None:
        """Move models back to CPU after a run finishes."""
        self.model.to(torch.device("cpu"))

    @abstractmethod
    def _infer_one_epoch(self, writer: PredictionWriter | None) -> None:
        """Run one inference epoch over the data loader.

        Args:
            writer: Prediction writer to accumulate results, or ``None`` to
                discard predictions.
        """
        ...
