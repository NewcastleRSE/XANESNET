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

import time

from xanesnet.models import Model

from .base import EarlyStopper
from .registry import EarlyStopperRegistry


@EarlyStopperRegistry.register("time")
class TimeStopper(EarlyStopper):
    """
    Early stopper that stops when a certain time limit is exceeded.
    """

    def __init__(
        self,
        early_stopper_type: str,
        restore_best: bool,
        time_limit_seconds: float,
    ) -> None:
        super().__init__(early_stopper_type, restore_best)

        self.time_limit_seconds = time_limit_seconds
        self.start_time = time.time()

    def step(self, value: float, model: Model, epoch: int) -> bool:
        _ = super().step(value, model, epoch)

        elapsed_time = time.time() - self.start_time
        return elapsed_time > self.time_limit_seconds
