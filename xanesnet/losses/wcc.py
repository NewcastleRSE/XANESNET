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
from torch import nn

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("wcc")
class WCCLoss(Loss):
    """
    Computes the weighted cross-correlation loss between y_pred and y_true based on the
    method proposed in [1].
    Args:
        gaussianHWHM: Scalar value for full-width-at-half-maximum of Gaussian weight function.
    Reference:
    [1] Källman, E., Delcey, M.G., Guo, M., Lindh, R. and Lundberg, M., 2020.
        "Quantifying similarity for spectra with a large number of overlapping transitions: Examples
        from soft X-ray spectroscopy." Chemical Physics, 535, p.110786.
    """

    def __init__(
        self,
        loss_type: str,
        gaussian_hwhm: int = 10,
    ) -> None:
        super().__init__(loss_type)

        self.gaussian_hwhm = gaussian_hwhm

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_features = targets.shape[1]
        n_samples = targets.shape[0]

        width2 = (self.gaussian_hwhm / np.sqrt(2.0 * np.log(2))) * 2

        corr = nn.functional.conv1d(targets.unsqueeze(0), preds.unsqueeze(1), padding="same", groups=n_samples)
        corr1 = nn.functional.conv1d(targets.unsqueeze(0), targets.unsqueeze(1), padding="same", groups=n_samples)
        corr2 = nn.functional.conv1d(preds.unsqueeze(0), preds.unsqueeze(1), padding="same", groups=n_samples)

        corr = corr.squeeze(0)
        corr1 = corr1.squeeze(0)
        corr2 = corr2.squeeze(0)

        dx = torch.ones(n_samples)
        de = ((n_features / 2 - torch.arange(0, n_features))[:, None] * dx[None, :]).T
        weight = np.exp(-de * de / (2 * width2))

        norm = torch.sum(corr * weight, 1)
        norm1 = torch.sum(corr1 * weight, 1)
        norm2 = torch.sum(corr2 * weight, 1)
        similarity = torch.clip(norm / torch.sqrt(norm1 * norm2), 0, 1)

        loss = 1 - torch.mean(similarity)
        return loss
