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

"""Abstract base class for all XANESNET loss functions."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class Loss(nn.Module, ABC):
    """Abstract base class for all XANESNET loss functions.

    Args:
        loss_type: Identifier string for the concrete loss type.
    """

    def __init__(
        self,
        loss_type: str,
    ) -> None:
        """Initialize ``Loss``."""
        super().__init__()

        self.loss_type = loss_type

    @abstractmethod
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the loss between predictions and targets.

        Args:
            preds: Model output predictions ``(B, N)``.
            targets: Ground-truth target values ``(B, N)``.
            reduction: ``"mean"`` returns the scalar loss; ``"none"`` returns
                the energy-resolved loss map with shape ``(B, N)`` where
                supported.

        Returns:
            Loss tensor.
        """
        ...
