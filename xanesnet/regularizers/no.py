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

"""No-op regularizer for XANESNET."""

import torch

from xanesnet.models import Model

from .base import Regularizer
from .registry import RegularizerRegistry


@RegularizerRegistry.register("no")
@RegularizerRegistry.register("none")
class NoReg(Regularizer):
    """No-op regularizer that always returns zero.

    Useful as a drop-in when regularization should be disabled while keeping
    the same interface.

    Args:
        regularizer_type: Identifier string for this regularizer type.
        weight: Unused; present for interface consistency. Defaults to ``1.0``.
    """

    def __init__(
        self,
        regularizer_type: str,
        weight: float = 1.0,
    ) -> None:
        """Initialize ``NoReg``."""
        super().__init__(regularizer_type, weight=weight)

    def forward(self, model: Model) -> torch.Tensor:
        """Return a scalar zero regularization term.

        Args:
            model: The model used only to infer the output device and dtype
                from its first parameter.

        Returns:
            Scalar zero tensor matching the first model parameter.
        """
        return next(model.parameters()).new_zeros(())
