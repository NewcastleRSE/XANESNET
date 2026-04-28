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

"""Multi-kernel SSIM loss for 1-D signals."""

import torch
import torch.nn.functional as F

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("mkssim1d")
class MultiKernel_SSIM_1D(Loss):
    """Multi-kernel SSIM loss for 1-D signals.

    Evaluates SSIM at several Gaussian kernel sizes, where each scale is
    determined as a fraction of the signal length ``N``. Scales can be
    combined multiplicatively (default) or via a weighted sum.

    Args:
        loss_type: Identifier string for this loss type.
        N: Length of the input signal.
        fractions: Kernel size fractions of ``N`` for each SSIM scale.
            Defaults to ``[0.01, 0.05, 0.10, 0.15, 0.2, 0.25]``.
        data_range: Dynamic range of the signal (``max - min``).
            Defaults to ``1.0``.
        K: Stability constants ``(K1, K2)`` for luminance and
            contrast-structure terms. Defaults to ``(0.01, 0.03)``.
        device: Compatibility argument retained for configuration parity.
            Kernels are moved to the device of ``preds`` during ``forward``.
            Defaults to ``'cpu'``.
        use_weighted_sum: If ``True``, combine scales via weighted sum
            instead of the default multiplicative combination.
            Defaults to ``False``.
        weights: Per-scale weights when ``use_weighted_sum=True``.
            If ``None``, uniform weights are used. Defaults to ``None``.
        final_combine: If ``True``, collapse scales into a single map.
            If ``False``, return a per-scale stack. Defaults to ``True``.
        final_mean: If ``True``, reduce the loss map to one value per batch
            element. Defaults to ``True``.
    """

    def __init__(
        self,
        loss_type: str,
        N: int,
        fractions: list[float] | tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.2, 0.25),
        data_range: float = 1.0,  # max - min
        K: tuple[float, float] = (0.01, 0.03),
        device: str | torch.device = "cpu",  # TODO do we still need this argument?
        use_weighted_sum: bool = False,
        weights: list[float] | None = None,
        final_combine: bool = True,
        final_mean: bool = True,
    ) -> None:
        """Initialize ``MultiKernel_SSIM_1D``."""
        super().__init__(loss_type)
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.N = N
        self.use_weighted_sum = use_weighted_sum
        self.final_combine = final_combine
        self.final_mean = final_mean

        # Get kernel sizes and sigmas
        self.kernel_sizes, self.gaussian_sigmas = self._get_kernel_sizes(fractions)

        # Create Gaussian masks for each kernel size
        g_masks = []
        for ks, sigma in zip(self.kernel_sizes, self.gaussian_sigmas):
            assert ks % 2 == 1, "Kernel size must be odd"
            assert ks.dtype == torch.long, "Kernel size must be integer"

            g = self._fspecial_gauss_1d(int(ks.item()), sigma.item())
            g = g.view(1, 1, -1)
            g_masks.append(g)
        self.g_masks = tuple(g_masks)
        self.device = device

        # Weights for weighted sum
        if use_weighted_sum:
            if weights is not None:
                if len(weights) != len(g_masks):
                    raise ValueError("Number of weights must match number of kernel sizes")
                self.weights = torch.tensor(weights, dtype=torch.float32)
            else:
                self.weights = torch.ones(len(g_masks), dtype=torch.float32) / len(g_masks)

    def _get_kernel_sizes(self, fractions: list[float] | tuple[float, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute odd kernel sizes and Gaussian sigmas from signal-length fractions.

        Args:
            fractions: Fraction of signal length ``N`` for each kernel scale.

        Returns:
            Tuple of ``(kernel_sizes, gaussian_sigmas)`` as long and float tensors.
        """

        def make_odd(x: torch.Tensor) -> torch.Tensor:
            """Return an odd window size derived from the input value."""
            x = torch.round(x).long()
            return x + (1 - x % 2)

        fractions_tensor = torch.tensor(fractions)
        kernel_sizes = make_odd(fractions_tensor * self.N)
        kernel_sizes = torch.where(fractions_tensor == 0.0, torch.ones_like(kernel_sizes), kernel_sizes)
        gaussian_sigmas = torch.clamp(kernel_sizes / 6.0, min=0.3)

        return kernel_sizes, gaussian_sigmas

    def _fspecial_gauss_1d(self, size: int, sigma: float) -> torch.Tensor:
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

    @staticmethod
    def _prepare_mask(mask: torch.Tensor, target_ndim: int) -> torch.Tensor:
        """Make a spectral mask rank-compatible with an unreduced loss map.

        Args:
            mask: Spectral mask with rank 2, 3, or 4. Rank-2 masks such as
                ``(B, N)`` are expanded by inserting singleton axes at
                dimension 1 until the rank matches ``target_ndim``. A rank-4
                mask is also accepted for a rank-3 target map when its third
                axis is singleton, e.g. ``(B, 1, 1, N)``.
            target_ndim: Expected rank of the unreduced loss map. In this
                module this is typically ``3`` for combined maps of shape
                ``(B, 1, N)`` or ``4`` for per-scale maps of shape
                ``(B, num_scales, 1, N)``.

        Returns:
            A mask tensor with rank ``target_ndim`` that is ready to be
            multiplied with the unreduced loss map.

        Raises:
            ValueError: If ``mask`` does not have rank 2, 3, or 4, or if it
                cannot be adapted to rank ``target_ndim`` without leaving an
                incompatible non-singleton axis.
        """
        if mask.ndim not in (2, 3, 4):
            raise ValueError(f"Expected mask to have rank 2, 3, or 4, got {mask.ndim}")

        while mask.ndim < target_ndim:
            mask = mask.unsqueeze(1)

        if mask.ndim == target_ndim + 1 and target_ndim == 3 and mask.shape[2] == 1:
            mask = mask.squeeze(2)

        if mask.ndim != target_ndim:
            raise ValueError(f"Expected mask to broadcast to a rank-{target_ndim} loss map, got rank {mask.ndim}")

        return mask

    # TODO what is the mask for?
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the multi-kernel 1-D SSIM loss.

        Args:
            preds: Predicted signals ``(B, N)``.
            targets: Ground-truth signals ``(B, N)``.
            mask: Optional mask tensor with shape ``(B, N)`` or broadcastable to
                the unreduced loss map. Defaults to ``None``.

        Returns:
            Scalar loss tensor when ``final_mean=True``. Otherwise returns the
            unreduced loss map with shape ``(B, N)`` when ``final_combine=True``
            or ``(B, num_scales, N)`` when ``final_combine=False``.
        """
        preds = preds.unsqueeze(1)
        targets = targets.unsqueeze(1)

        loss_scales = []

        for g in self.g_masks:
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

        if self.final_combine:
            if self.use_weighted_sum:
                # Stack tensors along a new dimension: shape [num_scales, B, 1, L]
                loss_stack = torch.stack(loss_scales, dim=0)
                weights = self.weights.to(device=preds.device, dtype=preds.dtype).view(-1, 1, 1, 1)
                loss_ms_ssim = torch.sum(weights * loss_stack, dim=0)
            else:
                # Multiplicative combination (default)
                loss_ms_ssim = torch.ones_like(loss_scales[0])
                for ls in loss_scales:
                    loss_ms_ssim *= ls
        else:
            loss_ms_ssim = torch.stack(loss_scales, dim=1)

        loss_ms_ssim = 1 - loss_ms_ssim

        if mask is not None:
            mask = self._prepare_mask(mask, loss_ms_ssim.ndim).to(device=loss_ms_ssim.device, dtype=loss_ms_ssim.dtype)
            loss_ms_ssim = loss_ms_ssim * mask

        if self.final_mean:
            if mask is not None:
                denom = mask.sum().clamp_min(torch.finfo(loss_ms_ssim.dtype).eps)
                return loss_ms_ssim.sum() / denom
            return torch.mean(loss_ms_ssim)

        if self.final_combine:
            return loss_ms_ssim.squeeze(1)
        return loss_ms_ssim.squeeze(2)
