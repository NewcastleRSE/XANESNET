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
from dataclasses import dataclass

import numpy as np
import torch

from typing import List, Optional, Tuple

from xanesnet.models.base_model import Model
from xanesnet.scheme.base_predict import Predict
from xanesnet.utils.fourier import inverse_fft
from xanesnet.utils.mode import Mode
from xanesnet.utils.gaussian import GaussianSynthesis


@dataclass
class Prediction:
    """Data class to hold prediction results, including mean and standard deviation."""

    xyz_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None


class NNPredict(Predict):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        self.fft = kwargs.get("fourier")
        self.gaussian = kwargs.get("gaussian")

    def predict(self, model):
        """Perform standard single-model prediction."""

        data_loader = self._create_loader(model, self.dataset)

        model.eval()
        predictions, targets = [], []

        synthesis = None
        if self.mode is Mode.XYZ_TO_XANES and self.gaussian:
            synthesis = GaussianSynthesis(
                basis=self.dataset.gauss_basis, nonneg_output=False
            )
            synthesis.eval()

        with torch.no_grad():
            for data in data_loader:
                # Pass X or batch object to model
                input_data = data if model.batch_flag else data.x
                output = model(input_data)
                # Inverse FFT transform
                if self.fft:
                    output = inverse_fft(output)
                # Gaussian synthesis
                if self.gaussian:
                    output = synthesis.forward_from_coeffs(output)

                output = self.to_numpy(output)
                predictions.append(output)

                if self.pred_eval:
                    target = self.to_numpy(data.y)
                    targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        if self.pred_eval:
            Predict.print_mse("target", "prediction", targets, predictions)

        return predictions, targets

    def predict_std(self, model: Model) -> Prediction:
        """Perform standard single-model prediction."""
        logging.info(
            f"\n--- Starting prediction with model: {model.__class__.__name__.lower()} ---"
        )

        predictions, targets = self.predict(model)
        # zero (dummy) standard deviation array.
        std_pred = np.zeros_like(predictions)

        if self.mode is Mode.XANES_TO_XYZ:
            return Prediction(xyz_pred=(predictions, std_pred))
        return Prediction(xanes_pred=(predictions, std_pred))

    def predict_bootstrap(self, model_list: List[Model]) -> Prediction:
        """Aggregate predictions from multiple bootstrap-trained models."""
        # Get all predictions and targets from model_list
        prediction_list, targets = self._predict_from_models(model_list)

        mean_pred = np.mean(prediction_list, axis=0)
        std_pred = np.std(prediction_list, axis=0)

        if self.pred_eval:
            logging.info("-" * 55)
            Predict.print_mse("target", "mean prediction", targets, mean_pred)

        if self.mode is Mode.XANES_TO_XYZ:
            return Prediction(xyz_pred=(mean_pred, std_pred))
        return Prediction(xanes_pred=(mean_pred, std_pred))

    def predict_ensemble(self, model_list: List[Model]) -> Prediction:
        """Aggregate predictions from an ensemble of models."""
        return self.predict_bootstrap(model_list)

    def _predict_from_models(self, model_list: List[Model]):
        """Predictions for a list of models."""
        prediction_list, targets = [], []

        for i, model in enumerate(model_list, start=0):
            logging.info(
                f">> Predicting with model {model.__class__.__name__.lower()} ({i}/{len(model_list)})..."
            )
            predictions, targets = self.predict(model)
            prediction_list.append(predictions)

        return np.array(prediction_list), targets
