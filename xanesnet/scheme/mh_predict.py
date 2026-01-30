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

import numpy as np
import torch


from xanesnet.scheme import NNPredict
from xanesnet.scheme.base_predict import Predict
from xanesnet.utils.fourier import fft_inverse
from xanesnet.utils.gaussian import gaussian_inverse


class MHPredict(NNPredict):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        self.mh_flag = 1

    def predict(self, model):
        """Perform standard single-model prediction."""

        data_loader = self._create_loader(model, self.dataset)

        model.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for data in data_loader:
                # Pass X or batch object to model
                input_data = data if model.batch_flag else data.x
                output = model(input_data).squeeze(1)

                if self.pred_eval:
                    # Select the prediction corresponding to head index
                    head_idx = data.head_idx
                    output = output[head_idx].squeeze(0)

                    target = self._postprocess(data.y)
                    targets.append(target)

                output = self._postprocess(output)
                predictions.append(output)

        predictions = np.array(predictions)
        targets = np.array(targets)

        if self.pred_eval:
            Predict.print_mse("target", "prediction", targets, predictions)

        return predictions, targets
