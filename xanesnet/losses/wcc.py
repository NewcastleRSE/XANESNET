# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

"""Weighted cross-correlation (WCC) loss for XANESNET."""

import torch
import torch.nn.functional as F

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("wcc")
class WCCLoss(Loss):
    """Weighted cross-correlation (WCC) loss.

    Computes the WCC similarity between predictions and targets using a
    Gaussian weight function centred at zero lag, following the method of
    Kallman et al. [1].

    References:
        [1] Kallman, E., Delcey, M.G., Guo, M., Lindh, R. and Lundberg, M.
            (2020). Quantifying similarity for spectra with a large number of
            overlapping transitions: Examples from soft X-ray spectroscopy.
            Chemical Physics, 535, p.110786.

    Args:
        loss_type: Identifier string for this loss type.
        gaussian_hwhm: Half-width at half-maximum of the Gaussian weight
            function in spectral bins. Defaults to ``10``.
    """

    def __init__(
        self,
        loss_type: str,
        gaussian_hwhm: int = 10,
    ) -> None:
        """Initialize ``WCCLoss``."""
        super().__init__(loss_type)

        if gaussian_hwhm <= 0:
            raise ValueError(f"gaussian_hwhm must be positive, got {gaussian_hwhm}")

        self.gaussian_hwhm = gaussian_hwhm

    @staticmethod
    def _centered_correlation(signals: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """Return the centered correlation window with the same width as the inputs."""
        n_features = signals.shape[1]
        full = F.conv1d(signals.unsqueeze(0), kernels.unsqueeze(1), padding=n_features - 1, groups=signals.shape[0])
        start = (full.shape[-1] - n_features) // 2
        return full.squeeze(0)[:, start : start + n_features]

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the WCC loss.

        Args:
            preds: Model output predictions ``(B, N)``.
            targets: Ground-truth spectral targets ``(B, N)``.

        Returns:
            Scalar loss tensor in ``[0, 1]``.
        """
        n_features = targets.shape[1]
        n_samples = targets.shape[0]
        dtype = preds.dtype
        device = preds.device

        width2 = (float(self.gaussian_hwhm) ** 2) / torch.log(torch.tensor(2.0, dtype=dtype, device=device))

        corr = self._centered_correlation(targets, preds)
        corr1 = self._centered_correlation(targets, targets)
        corr2 = self._centered_correlation(preds, preds)

        lag = torch.arange(0, n_features, dtype=dtype, device=device) - (n_features // 2)
        weight = torch.exp(-(lag * lag) / width2).expand(n_samples, -1)

        norm = torch.sum(corr * weight, 1)
        norm1 = torch.sum(corr1 * weight, 1)
        norm2 = torch.sum(corr2 * weight, 1)
        denom = torch.sqrt(torch.clamp(norm1 * norm2, min=torch.finfo(dtype).eps))
        similarity = torch.clamp(norm / denom, 0, 1)

        loss = 1 - torch.mean(similarity)
        return loss
