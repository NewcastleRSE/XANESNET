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


@LossRegistry.register("mwssim")
class MultiWindowSSIMLoss(Loss):
    def __init__(
        self,
        type: str,
        spec_size,
        weights,
        fractions,
    ):
        super().__init__(type)

        self.fractions = fractions
        self.weights = weights

        window_sizes, sigmas = self.compute_window_scales(spec_size)
        weights = [w / sum(weights) for w in weights]

        self.window_sizes = window_sizes
        self.sigmas = sigmas
        self.weights = weights

    def forward(self, y, pred):
        K = len(self.window_sizes)

        # Compute global data range once
        flat = torch.cat([pred.reshape(-1), y.reshape(-1)])
        data_range = (flat.max() - flat.min()).clamp(min=1e-12).item()

        if self.weights is None:
            weights = [1.0 / K] * K
        else:
            s = sum(self.weights)
            weights = [w / s for w in self.weights]

        per_scale_ssim = []

        for w_size, sigma in zip(self.window_sizes, self.sigmas):
            val = self.ssim_1d(
                pred,
                y,
                window_size=w_size,
                window_sigma=sigma,
                data_range=data_range,
            )
            per_scale_ssim.append(val)

        multi_ssim = sum(w * v for w, v in zip(weights, per_scale_ssim))
        multi_err = 1.0 - multi_ssim

        return multi_err

    def compute_window_scales(self, N):
        """
        N = length of spectrum
        Returns window_sizes and sigmas for:
        point-by-point, 10%, 25%, 33%, 50%, 100%.
        """

        def make_odd(x):
            """Convert x to the nearest odd integer ≥ 1."""
            x = int(round(x))
            if x < 1:
                return 1
            return x if x % 2 == 1 else x + 1

        window_sizes = []
        sigmas = []

        for frac in self.fractions:
            if frac == 0.0:
                w = 1  # point-by-point, cannot be 0
            else:
                w = make_odd(frac * N)

            sigma = max(0.3, w / 6.0)  # robust sigma rule
            window_sizes.append(w)
            sigmas.append(sigma)

        return window_sizes, sigmas

    def gaussian_window_1d(self, window_size: int, sigma: float, device=None, dtype=None):
        coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()
        return g.view(1, 1, -1)

    # -----------------------------
    # Single-scale SSIM
    # -----------------------------
    def ssim_1d(self, x, y, window_size=11, window_sigma=1.5, data_range=None):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            y = y.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)

        if data_range is None:
            flat = torch.cat([x.reshape(-1), y.reshape(-1)])
            data_range = (flat.max() - flat.min()).clamp(min=1e-12).item()

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        device = x.device
        dtype = x.dtype

        window = self.gaussian_window_1d(window_size, window_sigma, device=device, dtype=dtype)
        pad = window_size // 2

        mu_x = F.conv1d(x, window, padding=pad)
        mu_y = F.conv1d(y, window, padding=pad)

        mu_x2 = mu_x**2
        mu_y2 = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv1d(x * x, window, padding=pad) - mu_x2
        sigma_y2 = F.conv1d(y * y, window, padding=pad) - mu_y2
        sigma_xy = F.conv1d(x * y, window, padding=pad) - mu_xy

        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

        ssim_map = numerator / (denominator + 1e-12)
        return ssim_map.mean()
