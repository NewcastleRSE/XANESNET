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

"""Spectral loss with blurred coarse, detail, and gradient components."""

import math

import torch
import torch.nn.functional as F

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("specplus")
class SpectralLossPlus(Loss):
    """Multi-component spectral loss combining blurred, detail, and gradient terms.

    The total loss is ``alpha * Lc + beta * Ld + gamma * Lg`` where:

    - ``Lc`` — coarse MSE between Gaussian-blurred ``preds`` and ``targets``.
    - ``Ld`` — peak-aware weighted MSE on the residual fine detail.
    - ``Lg`` — Huber loss on first-order gradients (shape consistency).

    Args:
        loss_type: Identifier string for this loss type.
        blur_sigma_bins: Gaussian blur width in spectral bins for the coarse
            component. Defaults to ``5.0``.
        alpha: Weight for the coarse loss ``Lc``. Defaults to ``0.4``.
        beta: Weight for the detail loss ``Ld``. Defaults to ``0.6``.
        gamma: Weight for the gradient loss ``Lg``. Defaults to ``0.2``.
        huber_delta: Transition point for the Huber loss. Defaults to ``0.01``.
        kappa_peak: Amplitude of the peak-aware weight boost applied to ``Ld``.
            Defaults to ``0.15``.
    """

    def __init__(
        self,
        loss_type: str,
        blur_sigma_bins: float = 5.0,
        alpha: float = 0.4,
        beta: float = 0.6,
        gamma: float = 0.2,
        huber_delta: float = 0.01,
        kappa_peak: float = 0.15,
    ) -> None:
        super().__init__(loss_type)

        self.blur_sigma_bins = blur_sigma_bins
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.huber_delta = huber_delta
        self.kappa_peak = kappa_peak

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the spectral loss.

        Args:
            preds: Model output predictions ``(B, N)``.
            targets: Ground-truth spectral targets ``(B, N)``.
            mask: Unused; reserved for future use. Defaults to ``None``.

        Returns:
            Scalar loss tensor: ``alpha * Lc + beta * Ld + gamma * Lg``.
        """
        yb = self.gaussian_blur1d(targets, self.blur_sigma_bins)
        pb = self.gaussian_blur1d(preds, self.blur_sigma_bins)
        Lc = F.mse_loss(pb, yb)  # coarse (blurred) similarity

        # Peak-aware weighting
        w_peak = self.peak_weighting(targets, kappa=self.kappa_peak)
        diff_pred = preds - pb
        diff_true = targets - yb
        Ld = ((diff_pred - diff_true) ** 2 * w_peak).mean()  # weighted detail loss

        if preds.shape[-1] < 2:
            Lg = preds.new_zeros(())
        else:
            dy = preds[:, 1:] - preds[:, :-1]
            dyy = targets[:, 1:] - targets[:, :-1]
            Lg = self.huber_loss(dy, dyy, delta=self.huber_delta)  # gradient/shape consistency

        loss = self.alpha * Lc + self.beta * Ld + self.gamma * Lg

        # return L, (Lc, Ld, Lg)
        return loss

    def gaussian_blur1d(
        self,
        y: torch.Tensor,
        sigma_bins: float | torch.Tensor,
        k: int | None = None,
    ) -> torch.Tensor:
        """Blur a batch of 1-D signals with a Gaussian kernel.

        Args:
            y: Input signals ``(B, N)``.
            sigma_bins: Gaussian standard deviation in spectral bins. If a
                1-D tensor, only its maximum value is used.
            k: Kernel length. If ``None``, derived from ``sigma_bins`` as
                ``ceil(6 * sigma)`` rounded up to the nearest odd number.

        Returns:
            Blurred signals with the same shape as ``y``.

        Raises:
            ValueError: If ``k`` is not a positive odd integer.
        """
        y = y.contiguous()
        dtype = y.dtype
        device = y.device

        if y.shape[-1] <= 1:
            return y.clone()

        sigma = torch.as_tensor(sigma_bins, dtype=dtype, device=device)
        if sigma.ndim > 0:
            sigma = sigma.max()
        sigma = torch.clamp(sigma, min=torch.tensor(1e-6, dtype=dtype, device=device))

        if k is None:
            k = int(math.ceil(6.0 * float(sigma)))
            k = max(3, k | 1)
            k = min(k, 2 * y.shape[-1] - 1)
        elif k < 1 or k % 2 == 0:
            raise ValueError(f"k must be a positive odd integer, got {k}")

        half = k // 2
        grid = torch.arange(-half, half + 1, device=device, dtype=dtype)
        w = torch.exp(-0.5 * (grid / sigma) ** 2)
        w = w / (w.sum() + torch.finfo(dtype).eps)

        y1 = y.unsqueeze(1)  # (B, 1, N)
        ypad = F.pad(y1, (half, half), mode="reflect")
        out = F.conv1d(ypad, w.view(1, 1, -1))  # (B, 1, N)
        return out.squeeze(1)

    def huber_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        delta: float = 0.01,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the Huber (smooth L1) loss between two tensors.

        Args:
            x: First input tensor.
            y: Second input tensor (reference).
            delta: Transition point between quadratic and linear regions.
                Defaults to ``0.01``.
            reduction: Reduction mode — ``'mean'``, ``'sum'``, or ``'none'``.
                Defaults to ``'mean'``.

        Returns:
            Loss tensor (scalar for ``'mean'``/``'sum'``,
            element-wise for ``'none'``).

        Raises:
            ValueError: If ``reduction`` is not ``'mean'``, ``'sum'``, or ``'none'``.
        """
        delta_tensor = torch.as_tensor(delta, dtype=x.dtype, device=x.device)
        r = x - y
        abs_r = r.abs()
        quad = torch.minimum(abs_r, delta_tensor)
        lin = abs_r - quad
        loss = 0.5 * quad * quad + delta_tensor * lin
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {reduction}")

    def peak_weighting(self, y: torch.Tensor, kappa: float = 0.15) -> torch.Tensor:
        """Compute a peak-aware spatial weight map.

        Upweights strong intensities and concave regions to direct the detail
        loss towards spectrally important features.

        Args:
            y: Input signals ``(B, N)``.
            kappa: Amplitude of the weight boost applied at peaks and
                concavities. Defaults to ``0.15``.

        Returns:
            Weight map ``(B, N)`` with values ``>= 1``, detached from the
            computation graph.
        """
        # Normalize per spectrum
        y_norm = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True, unbiased=False) + 1e-6)
        w_amp = torch.sigmoid(y_norm)  # 0..1 stronger near peaks

        # Concavity: negative second derivative -> strong at peaks
        if y.shape[-1] < 3:
            concave = torch.zeros_like(y)
        else:
            d1 = y[:, 1:] - y[:, :-1]
            d2 = d1[:, 1:] - d1[:, :-1]  # (B, N-2)
            concave = F.relu(-d2)  # >0 for concave regions
            concave = F.pad(concave, (1, 1))  # align lengths

        # Combine and scale
        w = 1.0 + kappa * (w_amp + concave)
        return w.detach()  # detach so weights don't backpropagate
