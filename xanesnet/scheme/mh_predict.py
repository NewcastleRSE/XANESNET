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
from xanesnet.scheme import NNPredict
from xanesnet.scheme.base_predict import Predict
from xanesnet.utils.fourier import inverse_fft
from xanesnet.utils.mode import Mode
from xanesnet.utils.gaussian import SpectralPost


class MHPredict(NNPredict):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        self.mh_flag = 1

    def predict(self, model):
        """Perform standard single-model prediction."""

        data_loader = self._create_loader(model, self.dataset)

        model.eval()
        predictions, targets = [], []

        spectral_post = None
        if self.mode is Mode.XYZ_TO_XANES and self.gaussian:
            spectral_post = SpectralPost(basis=self.dataset.basis, nonneg_output=False)
            spectral_post.eval()

        with torch.no_grad():
            for data in data_loader:
                # Pass X or batch object to model
                input_data = data if model.batch_flag else data.x
                output = model(input_data).squeeze(1)

                # Inverse FFT transform
                if self.fft:
                    output = inverse_fft(output, self.fft_concat)
                # Gaussian reconstruction
                if self.gaussian:
                    output = spectral_post.forward_from_coeffs(output)

                if self.pred_eval:
                    # Select the prediction corresponding to head index (xanes)
                    head_idx = data.head_idx
                    output = output[head_idx].squeeze(0)

                    target = self.to_numpy(data.y)
                    targets.append(target)

                output = self.to_numpy(output)
                predictions.append(output)

        predictions = np.array(predictions)
        targets = np.array(targets)

        if self.pred_eval:
            Predict.print_mse("target", "prediction", targets, predictions)

        return predictions, targets
