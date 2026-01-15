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
from xanesnet.utils.fourier import inverse_fft
from xanesnet.utils.mode import Mode


@dataclass
class Prediction:
    """Data class to hold prediction results, including mean and standard deviation."""

    xyz_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xyz_recon: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_recon: Optional[Tuple[np.ndarray, np.ndarray]] = None


class AEGANPredict(Predict):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        self.fft = kwargs.get("fourier")

        self.recon_flag = 1

    def predict(self, model):
        """
        Performs a single prediction with a given model.
        """
        data_loader = self._create_loader(model, self.dataset)
        model.eval()

        predict_xyz, predict_xanes = [], []
        reconstruct_xyz, reconstruct_xanes = [], []
        targets_x, targets_y = [], []

        with torch.no_grad():
            for data in data_loader:
                # Prediction and reconstruction
                pred_xanes = recon_xanes = None

                targets_x.append(self.to_numpy(data.x))
                if self.pred_eval:
                    targets_y.append(self.to_numpy(data.y))

                if self.mode in [Mode.XYZ_TO_XANES, Mode.BIDIRECTIONAL]:
                    pred_xanes = model.predict_spectrum(
                        data if model.batch_flag else data.x
                    )
                    pred_xanes = self.to_numpy(pred_xanes)

                    recon_xyz = model.reconstruct_structure(
                        data if model.batch_flag else data.x
                    )
                    recon_xyz = self.to_numpy(recon_xyz)
                    # Append pred_xanes after fft check
                    reconstruct_xyz.append(recon_xyz)

                if self.mode in [Mode.XANES_TO_XYZ, Mode.BIDIRECTIONAL]:
                    input_data = data if model.batch_flag else data.y
                    pred_xyz = model.predict_structure(input_data)
                    pred_xyz = self.to_numpy(pred_xyz)

                    recon_xanes = model.reconstruct_spectrum(input_data)
                    recon_xanes = self.to_numpy(recon_xanes)
                    # Append recon_xanes after fft check
                    predict_xyz.append(pred_xyz)

                if self.fft and self.mode in [Mode.XYZ_TO_XANES, Mode.BIDIRECTIONAL]:
                    pred_xanes = inverse_fft(pred_xanes)
                if self.fft and self.mode in [Mode.XANES_TO_XYZ, Mode.BIDIRECTIONAL]:
                    recon_xanes = inverse_fft(recon_xanes)

                if pred_xanes is not None:
                    predict_xanes.append(pred_xanes)
                if recon_xanes is not None:
                    reconstruct_xanes.append(recon_xanes)

        predict_xyz = np.array(predict_xyz)
        predict_xanes = np.array(predict_xanes)
        reconstruct_xyz = np.array(reconstruct_xyz)
        reconstruct_xanes = np.array(reconstruct_xanes)
        targets_x = np.array(targets_x)
        targets_y = np.array(targets_y)

        if self.mode in [Mode.XYZ_TO_XANES, Mode.BIDIRECTIONAL]:
            Predict.print_mse(
                "XANES: target", "reconstruction", targets_x, reconstruct_xyz
            )
            if self.pred_eval:
                Predict.print_mse("XYZ: target", "prediction", targets_y, predict_xanes)
        if self.mode in [Mode.XANES_TO_XYZ, Mode.BIDIRECTIONAL]:
            Predict.print_mse(
                "XYZ: target", "reconstruction", targets_y, reconstruct_xanes
            )
            if self.pred_eval:
                Predict.print_mse("XANES: target", "prediction", targets_x, predict_xyz)

        return (
            predict_xyz,
            predict_xanes,
            reconstruct_xyz,
            reconstruct_xanes,
            targets_x,
            targets_y,
        )

    def predict_std(self, model: Model):
        """
        Performs a single prediction and returns the result with a zero (dummy)
        standard deviation array.
        """
        logging.info(
            f"\n--- Starting prediction with model: {model.__class__.__name__.lower()} ---"
        )

        # Get all predictions and reconstructions
        (
            predict_xyz,
            predict_xanes,
            reconstruct_xyz,
            reconstruct_xanes,
            _,
            _,
        ) = self.predict(model)

        if self.mode in [Mode.XANES_TO_XYZ]:
            return Prediction(
                xyz_pred=(predict_xyz, np.zeros_like(predict_xyz)),
                xanes_recon=(reconstruct_xanes, np.zeros_like(reconstruct_xanes)),
            )
        elif self.mode in [Mode.XYZ_TO_XANES]:
            return Prediction(
                xanes_pred=(predict_xanes, np.zeros_like(predict_xanes)),
                xyz_recon=(reconstruct_xyz, np.zeros_like(reconstruct_xyz)),
            )
        else:
            return Prediction(
                xyz_pred=(predict_xyz, np.zeros_like(predict_xyz)),
                xanes_pred=(predict_xanes, np.zeros_like(predict_xanes)),
                xyz_recon=(reconstruct_xyz, np.zeros_like(reconstruct_xyz)),
                xanes_recon=(reconstruct_xanes, np.zeros_like(reconstruct_xanes)),
            )

    def predict_bootstrap(self, model_list: List[Model]):
        """
        Predictions and reconstructions on multiple autoencoder models
        (bootstrapping) to calculate the mean and standard deviation.
        """
        # Get all predictions and reconstructions from model_list
        (
            predict_xyz_list,
            predict_xanes_list,
            reconstruct_xyz_list,
            reconstruct_xanes_list,
            targets_x,
            targets_y,
        ) = self._predict_from_models(model_list)

        mean_pred_xyz = std_pred_xyz = mean_pred_xanes = std_pred_xanes = None
        mean_recon_xanes = std_recon_xanes = mean_recon_xyz = std_recon_xyz = None

        logging.info("-" * 55)
        if self.mode in [Mode.XANES_TO_XYZ, Mode.BIDIRECTIONAL]:
            mean_pred_xyz = np.mean(predict_xyz_list, axis=0)
            std_pred_xyz = np.std(predict_xyz_list, axis=0)
            mean_recon_xanes = np.mean(reconstruct_xanes_list, axis=0)
            std_recon_xanes = np.std(reconstruct_xanes_list, axis=0)

            Predict.print_mse(
                "Mean XANES: target", "reconstruction", targets_y, mean_recon_xanes
            )
            if self.pred_eval:
                Predict.print_mse(
                    "Mean XYZ: target", "prediction", targets_x, mean_pred_xyz
                )

        if self.mode in [Mode.XYZ_TO_XANES, Mode.BIDIRECTIONAL]:
            mean_pred_xanes = np.mean(predict_xanes_list, axis=0)
            std_pred_xanes = np.std(predict_xanes_list, axis=0)
            mean_recon_xyz = np.mean(reconstruct_xyz_list, axis=0)
            std_recon_xyz = np.std(reconstruct_xyz_list, axis=0)

            Predict.print_mse(
                "Mean XYZ: target", "reconstruction", targets_x, mean_recon_xyz
            )
            if self.pred_eval:
                Predict.print_mse(
                    "Mean XANES: target", "prediction", targets_y, mean_pred_xanes
                )

        if self.mode in [Mode.XANES_TO_XYZ]:
            return Prediction(
                xyz_pred=(mean_pred_xyz, std_pred_xyz),
                xanes_recon=(mean_recon_xanes, std_recon_xanes),
            )
        elif self.mode in [Mode.XYZ_TO_XANES]:
            return Prediction(
                xanes_pred=(mean_pred_xanes, std_pred_xanes),
                xyz_recon=(mean_recon_xyz, std_recon_xyz),
            )
        else:
            return Prediction(
                xyz_pred=(mean_pred_xyz, np.zeros_like(std_pred_xyz)),
                xanes_pred=(mean_pred_xanes, np.zeros_like(mean_pred_xanes)),
                xyz_recon=(mean_recon_xyz, np.zeros_like(mean_recon_xyz)),
                xanes_recon=(mean_recon_xanes, np.zeros_like(mean_recon_xanes)),
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
        predict_xyz_list, predict_xanes_list = [], []
        reconstruct_xyz_list, reconstruct_xanes_list = [], []
        targets_x, targets_y = [], []

        for i, model in enumerate(model_list, start=1):
            logging.info(
                f">> Predicting with model {model.__class__.__name__.lower()} ({i}/{len(model_list)})..."
            )
            (
                predict_xyz,
                predict_xanes,
                reconstruct_xyz,
                reconstruct_xanes,
                targets_x,
                targets_y,
            ) = self.predict(model)

            if predict_xyz is not None:
                predict_xyz_list.append(predict_xyz)
            if predict_xanes is not None:
                predict_xanes_list.append(predict_xanes)
            if reconstruct_xyz is not None:
                reconstruct_xyz_list.append(reconstruct_xyz)
            if reconstruct_xanes is not None:
                reconstruct_xanes_list.append(reconstruct_xanes)

        predict_xyz_list = np.array(predict_xyz_list)
        predict_xanes_list = np.array(predict_xanes_list)
        reconstruct_xyz_list = np.array(reconstruct_xyz_list)
        reconstruct_xanes_list = np.array(reconstruct_xanes_list)

        return (
            predict_xyz_list,
            predict_xanes_list,
            reconstruct_xyz_list,
            reconstruct_xanes_list,
            targets_x,
            targets_y,
        )
