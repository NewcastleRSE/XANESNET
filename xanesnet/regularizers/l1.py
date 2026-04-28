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

"""L1 regularization for XANESNET."""

import torch

from xanesnet.models import Model

from .base import Regularizer
from .registry import RegularizerRegistry


@RegularizerRegistry.register("l1")
class L1Reg(Regularizer):
    """L1 regularization (sum of absolute parameter values).

    Args:
        regularizer_type: Identifier string for this regularizer type.
        weight: Scalar multiplier applied to the L1 penalty.
    """

    def __init__(
        self,
        regularizer_type: str,
        weight: float,
    ) -> None:
        """Initialize ``L1Reg``."""
        super().__init__(regularizer_type, weight)

    def forward(self, model: Model) -> torch.Tensor:
        """Compute the weighted L1 penalty over all model parameters.

        Args:
            model: The model whose parameters are penalised.

        Returns:
            Scalar L1 regularization loss tensor.
        """
        params = torch.cat([parameter.reshape(-1) for parameter in model.parameters()])
        return params.abs().sum() * self.weight
