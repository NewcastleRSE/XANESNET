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

"""Earth Mover's Distance (Wasserstein) loss for XANESNET."""

import torch

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("emd")
class EMDLoss(Loss):
    """Earth Mover's (Wasserstein) distance loss.

    Computes the discrete 1-D Earth Mover's Distance on a unit-spaced
    spectral grid as the L1 distance between the cumulative spectra of
    ``preds`` and ``targets``.

    Args:
        loss_type: Identifier string for this loss type.
    """

    def __init__(
        self,
        loss_type: str,
    ) -> None:
        super().__init__(loss_type)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Earth Mover's Distance loss.

        Args:
            preds: Model output predictions ``(B, N)``.
            targets: Ground-truth spectral targets ``(B, N)``.

        Returns:
            Scalar loss tensor summed over spectral bins and averaged over the
            batch dimension.
        """
        cdf_delta = torch.cumsum(preds - targets, dim=-1)
        return cdf_delta.abs().sum(dim=-1).mean()
