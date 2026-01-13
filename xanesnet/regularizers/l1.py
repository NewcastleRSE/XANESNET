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


@RegularizerRegistry.register("l1")
class L1Reg(Regularizer):
    """
    L1 regularization (sum of absolute parameter values)
    """

    def __init__(self, type: str):
        super().__init__(type)

    def forward(self, model):
        params = torch.cat([p.view(-1) for p in model.parameters()])
        return torch.norm(params, p=1)
