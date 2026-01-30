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
from typing import List, Tuple

import numpy as np

from abc import ABC, abstractmethod

import torch
import torch_geometric
from sklearn.metrics import mean_squared_error

from xanesnet.models.base_model import Model
from xanesnet.utils.fourier import fft_inverse
from xanesnet.utils.gaussian import gaussian_inverse
from xanesnet.utils.mode import Mode


class Predict(ABC):
    """Abstract base class defining the prediction interface for XANESNET models."""

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

        # Unpack parameters
        self.mode = kwargs.get("pred_mode")
        self.pred_eval = kwargs.get("pred_eval")

        self.fft = kwargs.get("fourier")
        self.gaussian = kwargs.get("gaussian")

        self.recon_flag = 0
        self.mh_flag = 0

    @abstractmethod
    def predict(self, model: Model):
        """
        Core forward prediction.

        Returns
        -------
        Tuple[np.array, np.array]
            A tuple containing Array of model output predictions,
            and Array of ground-truth target values
        """
        pass

    @abstractmethod
    def predict_std(self, model: Model):
        """
        Perform standard single-model prediction.

        Parameters
        ----------
        model : Model
            The trained model.

        Returns
        -------
        Prediction
            A `Prediction` dataclass object.
        """

        pass

    @abstractmethod
    def predict_bootstrap(self, model_list: List[Model]):
        """
        Aggregate predictions from multiple bootstrap-trained models.

        Parameters
        ----------
        model_list : List[Model]
            List of trained models.

        Returns
        -------
        Prediction
            A `Prediction` dataclass object.
        """

        pass

    @abstractmethod
    def predict_ensemble(self, model_list: List[Model]):
        """
        Aggregate predictions from an ensemble of models.


        Parameters
        ----------
        model_list : List[Model]
            List of trained models.

        Returns
        -------
        Prediction
            A `Prediction` dataclass object.
        """
        pass

    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        if self.mode == Mode.XYZ_TO_XANES or self.recon_flag:
            if self.fft:
                tensor = fft_inverse(tensor)
            if self.gaussian:
                tensor = gaussian_inverse(self.dataset.gauss_basis, tensor)

        return self._to_numpy(tensor)

    @staticmethod
    def _to_numpy(tensor):
        return tensor.squeeze().detach().cpu().numpy()

    @staticmethod
    def _create_loader(model, dataset):
        if model.gnn_flag:
            dataloader_cls = torch_geometric.data.DataLoader
        else:
            dataloader_cls = torch.utils.data.DataLoader

        return dataloader_cls(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def print_mse(
        source_name: str, target_name: str, data: np.ndarray, result: np.ndarray
    ):
        mse = mean_squared_error(data, result)
        logging.info(f"Mean Squared Error ({source_name} → {target_name}): {mse:.6f}")
