# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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
#
# Citations:
#   ...

"""Multi-scale SSIM loss."""

import torch
import torch.nn.functional as F

from .base import Loss
from .registry import LossRegistry

# TODO MSSSIM has the auto data_range feature. Note that currently when resuming
# TODO training from a checkpoint the running min/max is not restored, so the data_range will \
# TODO be re-inferred from the resumed training batches.


@LossRegistry.register("msssim")
class MultiScale_SSIM(Loss):
    """Multi-scale SSIM loss.

    Evaluates SSIM at several Gaussian kernel sizes, where each scale is
    determined as a fraction of the signal length.  The signal length is
    derived from ``input.shape[-1]`` on the first forward pass.

    Stability constants ``C1`` and ``C2`` depend on the signal's dynamic
    range.  When ``data_range`` is ``None`` (default) a running minimum
    and maximum of target values is tracked across forward passes.  The
    range starts at ``[0, 1]`` and expands whenever a batch contains
    values outside the current bounds, so it converges to the true data
    range within the first epoch and then stays fixed.

    Args:
        loss_type: Identifier string for this loss type.
        fractions: Kernel size fractions of the signal length for each SSIM
            scale.
        data_range: Dynamic range of the signal (``max - min``).  When
            ``None``, tracked as a running min/max across batches.  When a
            float, used directly.
        K: Stability constants ``(K1, K2)`` for luminance and
            contrast-structure terms.
        use_weighted_sum: If ``True``, combine scales via weighted sum
            instead of the default multiplicative combination.
        weights: Per-scale weights when ``use_weighted_sum=True``.
            If ``None``, uniform weights are used.
    """

    def __init__(
        self,
        loss_type: str,
        fractions: list[float] | tuple[float, ...],
        data_range: float | None,
        K: tuple[float, float],
        use_weighted_sum: bool,
        weights: list[float] | None,
    ) -> None:
        """Initialize ``MultiScale_SSIM``."""
        super().__init__(loss_type)
        self._K = K
        self._fractions = fractions
        self.use_weighted_sum = use_weighted_sum

        if data_range is not None:
            self.C1 = (K[0] * data_range) ** 2
            self.C2 = (K[1] * data_range) ** 2
            self._data_min = None
            self._data_max = None
        else:
            self._data_min = 0.0
            self._data_max = 1.0
            self.C1 = (K[0] * 1.0) ** 2
            self.C2 = (K[1] * 1.0) ** 2

        # Kernel masks are built lazily on the first forward pass.
        self._g_masks: tuple[torch.Tensor, ...] | None = None
        self._weights: torch.Tensor | None = None

        if use_weighted_sum:
            self._weights = self._build_weights(weights)

    def _build_kernels(self, N: int, fractions: list[float] | tuple[float, ...]) -> None:
        """Build Gaussian convolution kernels for the given signal length.

        Args:
            N: Signal length (number of spectral points).
            fractions: Kernel size fractions of *N*.
        """
        kernel_sizes, gaussian_sigmas = self._get_kernel_sizes(N, fractions)

        g_masks: list[torch.Tensor] = []
        for ks, sigma in zip(kernel_sizes, gaussian_sigmas):
            assert ks % 2 == 1, "Kernel size must be odd"
            assert ks.dtype == torch.long, "Kernel size must be integer"

            g = self._fspecial_gauss_1d(int(ks.item()), sigma.item())
            g = g.view(1, 1, -1)
            g_masks.append(g)
        self._g_masks = tuple(g_masks)

    def _build_weights(self, weights: list[float] | None) -> torch.Tensor:
        """Build per-scale weight tensor for weighted-sum combination.

        Args:
            weights: Per-scale weights.  Uniform when ``None``.

        Returns:
            Normalized weight tensor ``(num_scales,)``.
        """
        num_scales = len(self._fractions)
        if weights is not None:
            if len(weights) != num_scales:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of scales ({num_scales})")
            return torch.tensor(weights, dtype=torch.float32)
        return torch.ones(num_scales, dtype=torch.float32) / num_scales

    def _get_kernel_sizes(
        self, N: int, fractions: list[float] | tuple[float, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute odd kernel sizes and Gaussian sigmas from signal-length fractions.

        Args:
            N: Signal length in spectral points.
            fractions: Fraction of signal length *N* for each kernel scale.

        Returns:
            Tuple of ``(kernel_sizes, gaussian_sigmas)`` as long and float tensors.
        """

        def make_odd(x: torch.Tensor) -> torch.Tensor:
            """Return an odd window size derived from the input value."""
            x = torch.round(x).long()
            return x + (1 - x % 2)

        fractions_tensor = torch.tensor(fractions)
        kernel_sizes = make_odd(fractions_tensor * N)
        kernel_sizes = torch.where(fractions_tensor == 0.0, torch.ones_like(kernel_sizes), kernel_sizes)
        gaussian_sigmas = torch.clamp(kernel_sizes / 6.0, min=0.3)

        return kernel_sizes, gaussian_sigmas

    @staticmethod
    def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
        """Create a normalized 1-D Gaussian kernel.

        Args:
            size: Kernel length (number of points).
            sigma: Standard deviation of the Gaussian.

        Returns:
            Normalized Gaussian kernel ``(size,)``.
        """
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the multi-scale SSIM loss.

        Args:
            preds: Predicted signals ``(B, N)``.
            targets: Ground-truth signals ``(B, N)``.
            reduction: ``"mean"`` returns the scalar loss; ``"none"`` returns
                the combined energy-resolved loss map with shape ``(B, N)``.

        Returns:
            Loss tensor.

        Raises:
            ValueError: If ``reduction`` is neither ``"mean"`` nor ``"none"``.
        """
        if self._g_masks is None:
            N = preds.shape[-1]
            self._build_kernels(N, self._fractions)

        if self._data_min is not None:
            assert self._data_max is not None
            batch_min = targets.min().item()
            batch_max = targets.max().item()
            if batch_min < self._data_min or batch_max > self._data_max:
                self._data_min = min(self._data_min, batch_min)
                self._data_max = max(self._data_max, batch_max)
                dr = self._data_max - self._data_min
                self.C1 = (self._K[0] * dr) ** 2
                self.C2 = (self._K[1] * dr) ** 2

        g_masks = self._g_masks
        assert g_masks is not None

        preds = preds.unsqueeze(1)
        targets = targets.unsqueeze(1)

        loss_scales = []

        for g in g_masks:
            g = g.to(device=preds.device, dtype=preds.dtype)
            pad = g.shape[2] // 2

            # Means
            mux = F.conv1d(preds, g, padding=pad)
            muy = F.conv1d(targets, g, padding=pad)

            mux2 = mux**2
            muy2 = muy**2
            muxy = mux * muy

            # Variances / covariance
            sigmax2 = F.conv1d(preds * preds, g, padding=pad) - mux2
            sigmay2 = F.conv1d(targets * targets, g, padding=pad) - muy2
            sigmaxy = F.conv1d(preds * targets, g, padding=pad) - muxy

            l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)
            cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

            # combine luminance and contrast-structure per scale
            loss_scale = l * cs
            loss_scales.append(loss_scale)

        if self.use_weighted_sum:
            assert self._weights is not None
            # Stack tensors along a new dimension: shape [num_scales, B, 1, L]
            loss_stack = torch.stack(loss_scales, dim=0)
            weights = self._weights.to(device=preds.device, dtype=preds.dtype).view(-1, 1, 1, 1)
            loss_ms_ssim = torch.sum(weights * loss_stack, dim=0)
        else:
            # Multiplicative combination (default)
            loss_ms_ssim = torch.ones_like(loss_scales[0])
            for ls in loss_scales:
                loss_ms_ssim *= ls

        loss_map = 1 - loss_ms_ssim  # (B, 1, N)

        if reduction == "none":
            return loss_map.squeeze(1)
        if reduction == "mean":
            return torch.mean(loss_map)
        raise ValueError(f"Unsupported reduction: {reduction}")
