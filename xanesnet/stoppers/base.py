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

from abc import ABC, abstractmethod

from xanesnet.models import Model


class EarlyStopper(ABC):
    def __init__(
        self,
        stopper_type: str,
        restore_best: bool,
    ) -> None:
        self.stopper_type = stopper_type
        self.restore_best = restore_best

        self.best_state: dict | None = None
        self.best_value: float = float("inf")
        self.best_epoch: int = -1

    @abstractmethod
    def step(self, value: float | None, model: Model, epoch: int) -> bool: ...

    def restore(self, model: Model) -> tuple[float | None, int | None]:
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
            return self.best_value, self.best_epoch

        return None, None
