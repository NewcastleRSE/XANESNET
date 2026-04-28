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

"""Mean squared error loss for XANESNET."""

import torch
from torch import nn

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("mse")
class MSELoss(Loss):
    """Mean squared error loss.

    Args:
        loss_type: Identifier string for this loss type.
    """

    def __init__(
        self,
        loss_type: str,
    ) -> None:
        super().__init__(loss_type)

        self.loss = nn.MSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the mean squared error loss.

        Args:
            preds: Model output predictions ``(B, N)``.
            targets: Ground-truth target values ``(B, N)``.

        Returns:
            Scalar loss tensor.
        """
        return self.loss(preds, targets)
