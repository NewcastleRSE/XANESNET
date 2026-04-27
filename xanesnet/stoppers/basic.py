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


@EarlyStopperRegistry.register("basic")
class BasicStopper(EarlyStopper):
    """
    Early stopper that stops when no improvement is seen for a number of epochs.
    """

    def __init__(
        self,
        early_stopper_type: str,
        restore_best: bool,
        patience: int,
        min_delta: float = 0.0,
    ) -> None:
        super().__init__(early_stopper_type, restore_best)

        self.patience = patience
        self.min_delta = min_delta
        self.last_improvement_epoch: int | None = None

    def step(self, value: float, model: Model, epoch: int) -> bool:
        prev_best_value = self.best_value
        prev_best_epoch = self.best_epoch

        _ = super().step(value, model, epoch)

        if prev_best_epoch < 0:  # first call
            self.last_improvement_epoch = epoch
            return False

        if self.last_improvement_epoch is None:  # if not set yet but not first call
            self.last_improvement_epoch = prev_best_epoch

        epochs_since_prev_best = max(1, epoch - prev_best_epoch)
        required_total_delta = self.min_delta * epochs_since_prev_best
        meaningful_improvement = value < prev_best_value - required_total_delta

        if meaningful_improvement:
            self.last_improvement_epoch = epoch

        last_improvement = self.last_improvement_epoch
        epochs_without_improvement = epoch - last_improvement

        return epochs_without_improvement >= self.patience
