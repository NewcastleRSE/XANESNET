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


@LossRegistry.register("specplus")
class SpectralLossPlus(Loss):
    def __init__(
        self,
        type: str,
        blur_sigma_bins: float = 5.0,
        alpha: float = 0.4,
        beta: float = 0.6,
        gamma: float = 0.2,
        huber_delta: float = 0.01,
        kappa_peak: float = 0.15,
    ):
        super().__init__(type)
        self.blur_sigma_bins = blur_sigma_bins
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.huber_delta = huber_delta
        self.kappa_peak = kappa_peak

    def forward(self, y, pred):
        """
        Returns:
          L: scalar = alpha*Lc + beta*Ld + gamma*Lg
          (Lc, Ld, Lg): individual component scalars for logging
        Adds peak-aware weighting to Ld.
        """
        yb = self.gaussian_blur1d(y, self.blur_sigma_bins)
        pb = self.gaussian_blur1d(pred, self.blur_sigma_bins)
        Lc = F.mse_loss(pb, yb)  # coarse (blurred) similarity

        # Peak-aware weighting
        w_peak = self.peak_weighting(y, kappa=self.kappa_peak)
        diff_pred = pred - pb
        diff_true = y - yb
        Ld = ((diff_pred - diff_true) ** 2 * w_peak).mean()  # weighted detail loss

        dy = pred[:, 1:] - pred[:, :-1]
        dyy = y[:, 1:] - y[:, :-1]
        Lg = self.huber_loss(dy, dyy, delta=self.huber_delta)  # gradient/shape consistency

        loss = self.alpha * Lc + self.beta * Ld + self.gamma * Lg

        # return L, (Lc, Ld, Lg)
        return loss

    def gaussian_blur1d(self, y: torch.Tensor, sigma_bins, k: int = None) -> torch.Tensor:
        """
        y: (B, N) float tensor
        sigma_bins: float or 0-d/1-d tensor; if 1-d, only its max is used.
        """
        y = y.contiguous()
        dtype = y.dtype
        device = y.device

        sigma = torch.as_tensor(sigma_bins, dtype=dtype, device=device)
        if sigma.ndim > 0:
            sigma = sigma.max()
        sigma = torch.clamp(sigma, min=torch.tensor(1e-6, dtype=dtype, device=device))

        if k is None:
            k = int(math.ceil(6.0 * float(sigma)))
            k = max(3, k | 1)

        half = k // 2
        grid = torch.arange(-half, half + 1, device=device, dtype=dtype)
        w = torch.exp(-0.5 * (grid / sigma) ** 2)
        w = w / (w.sum() + torch.finfo(dtype).eps)

        y1 = y.unsqueeze(1)  # (B, 1, N)
        ypad = F.pad(y1, (half, half), mode="reflect")
        out = F.conv1d(ypad, w.view(1, 1, -1))  # (B, 1, N)
        return out.squeeze(1)

    def huber_loss(self, x, y, delta=0.01, reduction="mean"):
        delta = torch.as_tensor(delta, dtype=x.dtype, device=x.device)
        r = x - y
        abs_r = r.abs()
        quad = torch.minimum(abs_r, delta)
        lin = abs_r - quad
        loss = 0.5 * quad * quad + delta * lin
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    def peak_weighting(self, y: torch.Tensor, kappa: float = 0.15) -> torch.Tensor:
        """
        Peak-aware weighting map in [1, 1+2*kappa]:
          - Upweights strong intensities and concave peaks.
          - y: (B, N)
        """
        # Normalize per spectrum
        y_norm = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-6)
        w_amp = torch.sigmoid(y_norm)  # 0..1 stronger near peaks

        # Concavity: negative second derivative -> strong at peaks
        d1 = y[:, 1:] - y[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]  # (B, N-2)
        concave = F.relu(-d2)  # >0 for concave regions
        concave = F.pad(concave, (1, 1))  # align lengths

        # Combine and scale
        w = 1.0 + kappa * (w_amp + concave)
        return w.detach()  # detach so weights don't backpropagate
