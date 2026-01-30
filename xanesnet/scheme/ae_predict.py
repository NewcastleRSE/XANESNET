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
import numpy as np
import torch

from typing import Optional, Tuple, List
from dataclasses import dataclass

from xanesnet.models.base_model import Model
from xanesnet.scheme.base_predict import Predict
from xanesnet.utils.fourier import fft_inverse
from xanesnet.utils.mode import Mode


@dataclass
class Prediction:
    """Data class to hold prediction results, including mean and standard deviation."""

    xyz_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xyz_recon: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_recon: Optional[Tuple[np.ndarray, np.ndarray]] = None


class AEPredict(Predict):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        self.recon_flag = 1

    def predict(self, model):
        """
        Performs a single prediction with a given model.
        """
        data_loader = self._create_loader(model, self.dataset)

        model.eval()
        predictions, reconstructions, targets_x, targets_y = [], [], [], []

        with torch.no_grad():
            for data in data_loader:
                # Pass X or batch object to model
                input_data = data if model.batch_flag else data.x

                pred = model.predict(input_data)
                pred = self._postprocess(pred)

                recon = model.reconstruct(input_data)
                recon = self._postprocess(recon)

                predictions.append(pred)
                reconstructions.append(recon)

                x = self._postprocess(data.x)
                targets_x.append(x)

                if self.pred_eval:
                    y = self._postprocess(data.y)
                    targets_y.append(y)

        predictions = np.array(predictions)
        reconstructions = np.array(reconstructions)
        targets_x = np.array(targets_x)
        targets_y = np.array(targets_y)

        if self.mode == Mode.XANES_TO_XYZ:
            Predict.print_mse(
                "XANES target", "reconstruction", targets_x, reconstructions
            )
            if self.pred_eval:
                Predict.print_mse("XYZ target", "prediction", targets_y, predictions)
        elif self.mode == Mode.XYZ_TO_XANES:
            Predict.print_mse(
                "XYZ target", "reconstruction", targets_x, reconstructions
            )
            if self.pred_eval:
                Predict.print_mse("XANES target", "prediction", targets_y, predictions)

        return predictions, reconstructions, targets_x, targets_y

    def predict_std(self, model):
        """
        Performs a single prediction and returns the result with a zero (dummy)
        standard deviation array.
        """
        logging.info(
            f"\n--- Starting prediction with model: {model.__class__.__name__.lower()} ---"
        )

        # Get all predictions and reconstructions
        predictions, reconstructions, _, _ = self.predict(model)

        # Create dummy array for STD
        std_pred = np.zeros_like(predictions)
        std_recon = np.zeros_like(reconstructions)

        if self.mode is Mode.XANES_TO_XYZ:
            return Prediction(
                xyz_pred=(predictions, std_pred),
                xanes_recon=(reconstructions, std_recon),
            )

        return Prediction(
            xanes_pred=(predictions, std_pred), xyz_recon=(reconstructions, std_recon)
        )

    def predict_bootstrap(self, model_list: List[Model]):
        """
        Predictions and reconstructions on multiple autoencoder models
        (bootstrapping) to calculate the mean and standard deviation.
        """
        # Get all predictions and reconstructions from model_list
        pred_list, recon_list, targets_x, target_y = self._predict_from_models(
            model_list
        )

        # Calculate mean and std
        mean_pred = np.mean(pred_list, axis=0)
        std_pred = np.std(pred_list, axis=0)
        mean_recon = np.mean(recon_list, axis=0)
        std_recon = np.std(recon_list, axis=0)

        logging.info("-" * 55)
        Predict.print_mse("xanes", "mean reconstruction", targets_x, mean_recon)
        if self.pred_eval:
            Predict.print_mse("target", "mean prediction", target_y, mean_pred)

        if self.mode is Mode.XANES_TO_XYZ:
            return Prediction(
                xyz_pred=(mean_pred, std_pred), xanes_recon=(mean_recon, std_recon)
            )
        # predict_xanes
        return Prediction(
            xanes_pred=(mean_pred, std_pred), xyz_recon=(mean_recon, std_recon)
        )

    def predict_ensemble(self, model_list):
        """
        Performs predictions on an ensemble of models.
        Same to bootstrap
        """
        return self.predict_bootstrap(model_list)

    def _predict_from_models(self, model_list: List[Model]):
        """
        Predictions for a list of models.
        """
        pred_list, recon_list, targets_x, targets_y = [], [], [], []

        for i, model in enumerate(model_list, start=1):
            logging.info(
                f">> Predicting with model {model.__class__.__name__.lower()} ({i}/{len(model_list)})..."
            )
            pred, recon, targets_x, targets_y = self.predict(model)

            pred_list.append(pred)
            recon_list.append(recon)

        return np.array(pred_list), np.array(recon_list), targets_x, targets_y
