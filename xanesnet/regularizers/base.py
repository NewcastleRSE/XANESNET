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

"""Abstract base class for all XANESNET regularizers."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from xanesnet.models import Model


class Regularizer(nn.Module, ABC):
    """Abstract base class for all XANESNET regularizers.

    Args:
        regularizer_type: Identifier string for the concrete regularizer type.
        weight: Scalar multiplier applied to the regularization term.
    """

    def __init__(
        self,
        regularizer_type: str,
        weight: float,
    ) -> None:
        super().__init__()

        self.regularizer_type = regularizer_type
        self.weight = weight

    @abstractmethod
    def forward(self, model: Model) -> torch.Tensor:
        """Compute the regularization penalty for a model.

        Args:
            model: The model whose parameters are penalised.

        Returns:
            Scalar regularization loss tensor.
        """
        ...
