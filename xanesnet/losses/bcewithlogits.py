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

from torch import nn

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("bce")
class BCEWithLogitsLoss(Loss):
    """
    Computes binary cross-entropy loss with logits
    """

    def __init__(
        self,
        loss_type: str,
    ):
        super().__init__(loss_type)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.loss(preds, targets)
