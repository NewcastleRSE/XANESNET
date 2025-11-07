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

from abc import ABC, abstractmethod

import torch
import torch_geometric
from sklearn.metrics import mean_squared_error


class Predict(ABC):
    """
    Base class for prediction

    """

    def __init__(self, dataset, mode, **kwargs):
        self.dataset = dataset

        self.mode = kwargs.get("pred_mode")
        self.pred_eval = kwargs.get("pred_eval")
        self.scaler = kwargs.get("scaler")
        self.fft = kwargs.get("fourier")
        self.fft_concat = kwargs.get("fourier_concat")
        self.gaussian = kwargs.get("gaussian")
        self.widths_eV = kwargs.get("widths_eV")
        self.basis_stride = kwargs.get("basis_stride")

        self.recon_flag = 0

    @abstractmethod
    def predict(self, model):
        pass

    @abstractmethod
    def predict_std(self, model):
        pass

    @abstractmethod
    def predict_bootstrap(self, model_list):
        pass

    @abstractmethod
    def predict_ensemble(self, model_list):
        pass

    @staticmethod
    def to_numpy(tensor):
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

    @staticmethod
    def setup_scaler(scaler, x_data, inverse: bool):
        if not inverse:
            scaler.fit(x_data)
            x_data = scaler.transform(x_data)
        else:
            x_data = scaler.inverse_transform(x_data)

        return x_data
