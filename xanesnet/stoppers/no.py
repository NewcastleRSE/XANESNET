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
    def __init__(
        self,
        early_stopper_type: str,
        restore_best: bool,
    ) -> None:
        super().__init__(early_stopper_type, restore_best)

    def step(self, value: float | None, model: Model, epoch: int) -> bool:
        if value is not None and value < self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return False
