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

"""No-op early stopper that never halts training."""

from xanesnet.models import Model

from .base import EarlyStopper
from .registry import EarlyStopperRegistry


@EarlyStopperRegistry.register("none")
@EarlyStopperRegistry.register("no")
class NoStopper(EarlyStopper):
    """No-op early stopper that never halts training.

    Still tracks the best metric value and optionally snapshots model
    weights (when ``restore_best`` is ``True``) so that ``restore`` can
    be used after training completes.
    """

    def __init__(
        self,
        early_stopper_type: str,
        restore_best: bool,
    ) -> None:
        """Initialise the no-op stopper.

        Args:
            early_stopper_type: Registry key identifying this stopper type.
            restore_best: Passed through to ``EarlyStopper``; controls
                whether model weights are snapshotted on improvement.
        """
        super().__init__(early_stopper_type, restore_best)

    def step(self, value: float, model: Model, epoch: int) -> bool:
        """Update tracking state and always return ``False``.

        Args:
            value: Current metric value to minimise.
            model: The model being trained.
            epoch: Current epoch index.

        Returns:
            Always ``False``.
        """
        _ = super().step(value, model, epoch)
        return False
