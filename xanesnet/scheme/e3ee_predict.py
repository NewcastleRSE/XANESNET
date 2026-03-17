"""
Custom prediction scheme for absorber-centred e3nn XANES models.
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch

from xanesnet.models.base_model import Model
from xanesnet.scheme.base_predict import Predict
from xanesnet.utils.mode import Mode


@dataclass
class Prediction:
    """Data class to hold prediction results, including mean and standard deviation."""

    xyz_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    targets: Optional[np.ndarray] = None


# -------------------------------------------------
# Prediction scheme
# -------------------------------------------------
class E3EEPredict(Predict):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def predict(self, model):
        """Perform standard single-model prediction."""

        data_loader = self._create_loader(model, self.dataset)

        model.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for data in data_loader:
                input_data = data if model.batch_flag else data.x

                output = model(input_data)
                output = self._postprocess(output)
                predictions.append(output)

                if self.pred_eval:
                    target = self._postprocess(data.y)
                    targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Flatten (n_batches, batch_size, nE) → (n_samples, nE)
        if predictions.ndim == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])

        if self.pred_eval and targets.ndim == 3:
            targets = targets.reshape(-1, targets.shape[-1])

        if self.pred_eval:
            Predict.print_mse("target", "prediction", targets, predictions)

        return predictions, targets

    def predict_std(self, model: Model) -> Prediction:
        """Perform standard single-model prediction."""
        logging.info(
            f"\n--- Starting prediction with model: {model.__class__.__name__.lower()} ---"
        )

        predictions, targets = self.predict(model)
        std_pred = np.zeros_like(predictions)

        if self.mode is Mode.XANES_TO_XYZ:
            return Prediction(xyz_pred=(predictions, std_pred), targets=targets)
        return Prediction(xanes_pred=(predictions, std_pred), targets=targets)

    def predict_bootstrap(self, model_list: List[Model]) -> Prediction:
        """Aggregate predictions from multiple bootstrap-trained models."""
        prediction_list, targets = self._predict_from_models(model_list)

        mean_pred = np.mean(prediction_list, axis=0)
        std_pred = np.std(prediction_list, axis=0)

        if self.pred_eval:
            logging.info("-" * 55)
            Predict.print_mse("target", "mean prediction", targets, mean_pred)

        if self.mode is Mode.XANES_TO_XYZ:
            return Prediction(xyz_pred=(mean_pred, std_pred), targets=targets)
        return Prediction(xanes_pred=(mean_pred, std_pred), targets=targets)

    def predict_ensemble(self, model_list: List[Model]) -> Prediction:
        """Aggregate predictions from an ensemble of models."""
        return self.predict_bootstrap(model_list)

    def _predict_from_models(self, model_list: List[Model]):
        """Predictions for a list of models."""
        prediction_list, targets = [], []

        for i, model in enumerate(model_list, start=0):
            logging.info(
                f">> Predicting with model {model.__class__.__name__.lower()} ({i+1}/{len(model_list)})..."
            )
            predictions, targets = self.predict(model)
            prediction_list.append(predictions)

        return np.array(prediction_list), targets
