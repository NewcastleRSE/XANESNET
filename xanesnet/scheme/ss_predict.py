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

import xanesnet
from xanesnet.models.base_model import Model
from xanesnet.scheme.base_predict import Predict
from xanesnet.utils.fourier import inverse_fft
from xanesnet.utils.mode import Mode


@dataclass
class Prediction:
    """Data class to hold prediction results, including mean and standard deviation."""

    xyz_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None


class SSPredict(Predict):
    def __init__(self, dataset, mode, **kwargs):
        super().__init__(dataset, mode, **kwargs)
        self.stride = kwargs.get("basis_stride")

    def predict(self, model):
        """
        Performs a single prediction with a given model.
        """
        from xanesnet.models.softshell import SpectralPost, SpectralBasis

        model.eval()
        predictions, targets = [], []
        data_loader = self._create_loader(model, self.dataset)

        # Initialise parameter-free spectral "post" component
        eV = self.dataset[0].e
        widths_bins = self.compute_widths_bins(eV)

        basis_eval = SpectralBasis(
            energies=eV,
            widths_bins=widths_bins,
            normalize_atoms=True,
            stride=self.stride,
        )

        spectral_post = SpectralPost(basis=basis_eval, nonneg_output=False)
        spectral_post.eval()

        # ---- Run inference ----
        with torch.no_grad():
            for data in data_loader:
                c_pred = model(data)
                output = spectral_post.forward_from_coeffs(c_pred)  # (B,N)
                output = self.to_numpy(output.squeeze(0))

                predictions.append(output)

                if self.pred_eval:
                    target = self.to_numpy(data.y)
                    targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        # ---- Evaluation ----
        if self.pred_eval:
            Predict.print_mse("target", "prediction", targets, predictions)

        return predictions, targets

    def predict_std(self, model: Model) -> Prediction:
        """
        Performs a single prediction and returns the result with a zero (dummy)
        standard deviation array.
        """
        logging.info(
            f"\n--- Starting prediction with model: {model.__class__.__name__.lower()} ---"
        )

        predictions, targets = self.predict(model)
        std_pred = np.zeros_like(predictions)

        return Prediction(xanes_pred=(predictions, std_pred))

    def predict_bootstrap(self, model_list: List[Model]) -> Prediction:
        """
        Performs predictions on multiple models (bootstrapping)
        """
        pass

    def predict_ensemble(self, model_list: List[Model]) -> Prediction:
        """
        Performs predictions on multiple models (ensemble)
        """
        pass

    @staticmethod
    def compute_widths_bins(eV):
        dE = float(eV[1] - eV[0])
        widths_eV = (0.5, 1.0, 2.0, 4.0)
        widths_bins = tuple(max(w / dE, 0.5) for w in widths_eV)

        return widths_bins
