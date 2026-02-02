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

from xanesnet.models import Model

from .base import EarlyStopper
from .registry import EarlyStopperRegistry


@EarlyStopperRegistry.register("none")
@EarlyStopperRegistry.register("no")
class NoStopper(EarlyStopper):
    """
    No-op early stopper that never stops training.
    """

    def __init__(
        self,
        early_stopper_type: str,
        restore_best: bool,
    ) -> None:
        super().__init__(early_stopper_type, restore_best)

    def step(self, value: float, model: Model, epoch: int) -> bool:
        _ = super().step(value, model, epoch)
        return False
