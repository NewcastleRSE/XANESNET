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

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("emd")
class EMDLoss(Loss):
    """
    Computes the Earth Mover (Wasserstein) distance
    """

    def __init__(self, type: str):
        super().__init__(type)

    def forward(self, preds, targets):
        return torch.mean(
            torch.square(torch.cumsum(targets, dim=-1) - torch.cumsum(preds, dim=-1)),
            dim=-1,
        ).sum()
