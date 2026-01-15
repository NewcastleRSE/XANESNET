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

from .base import Regularizer
from .registry import RegularizerRegistry


@RegularizerRegistry.register("no")
@RegularizerRegistry.register("none")
class NoReg(Regularizer):
    """
    No regularization
    """

    def __init__(
        self,
        regularizer_type: str,
    ):
        super().__init__(regularizer_type)

    def forward(self, model):
        return torch.tensor(0.0, device=next(model.parameters()).device)
