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

import torch
import torch.nn.functional as F

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("mkssim1d")
class MultiKernel_SSIM_1D(Loss):
    """
    Multi-kernel SSIM for 1D signals using different kernel sizes with appropriate sigmas
    """

    def __init__(
        self,
        loss_type: str,
        N: int,
        fractions: list[float] = [0.01, 0.05, 0.10, 0.15, 0.2, 0.25],
        data_range: float = 1.0,  # max - min
        K: tuple[float, float] = (0.01, 0.03),
        device: str | torch.device = "cpu",
        use_weighted_sum: bool = False,
        weights: list[float] | None = None,
        final_combine: bool = True,
        final_mean: bool = True,
    ) -> None:
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
        self.g_masks = g_masks
        self.device = device

        # Weights for weighted sum
        if use_weighted_sum:
            if weights is not None:
                assert len(weights) == len(g_masks), "Number of weights must match number of kernel sizes"
                self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
            else:
                self.weights = torch.ones(len(g_masks), dtype=torch.float32, device=device) / len(g_masks)

    def _get_kernel_sizes(self, fractions: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        def make_odd(x: torch.Tensor) -> torch.Tensor:
            x = torch.round(x).long()
            return x + (1 - x % 2)

        fractions_tensor = torch.tensor(fractions)
        kernel_sizes = make_odd(fractions_tensor * self.N)
        kernel_sizes = torch.where(fractions_tensor == 0.0, torch.ones_like(kernel_sizes), kernel_sizes)
        gaussian_sigmas = torch.clamp(kernel_sizes / 6.0, min=0.3)

        return kernel_sizes, gaussian_sigmas

    def _fspecial_gauss_1d(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask=None) -> torch.Tensor:
        """
        preds, targets: [B, 1, L] tensor
        """
        loss_scales = []

        for g in self.g_masks:
            g = g.to(self.device)
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
                weights = self.weights.view(-1, 1, 1, 1)  # broadcast weights
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
            loss_ms_ssim = loss_ms_ssim * mask

        if self.final_mean:
            if self.final_combine:
                return torch.mean(loss_ms_ssim, dim=[1, 2])
            else:
                return torch.mean(loss_ms_ssim, dim=[2, 3])
        else:
            return loss_ms_ssim
