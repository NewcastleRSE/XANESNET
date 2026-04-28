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

"""Abstract base class for XANESNET early-stopping strategies."""

from abc import ABC, abstractmethod

from xanesnet.models import Model


class EarlyStopper(ABC):
    """Abstract base class for early-stopping strategies.

    Tracks the best observed metric value and optionally snapshots the
    corresponding model weights so they can be restored at the end of
    training.
    """

    def __init__(
        self,
        early_stopper_type: str,
        restore_best: bool,
    ) -> None:
        """Initialise shared early-stopping state.

        Args:
            early_stopper_type: Registry key identifying this stopper type.
            restore_best: If ``True``, save a copy of the model weights
                whenever a new best value is observed so they can be
                restored later via ``restore``.
        """
        self.early_stopper_type = early_stopper_type
        self.restore_best = restore_best

        self.best_state: dict | None = None
        self.best_value: float = float("inf")
        self.best_epoch: int = -1

    @abstractmethod
    def step(self, value: float, model: Model, epoch: int) -> bool:
        """Evaluate whether training should stop after observing ``value``.

        The base implementation updates ``best_value``, ``best_epoch``, and
        (when ``restore_best`` is ``True``) ``best_state`` whenever a new
        minimum is found. Subclasses can call ``super().step()`` whenever they
        decide that the observed value should be recorded as the new best,
        then apply their own stopping criterion.

        Args:
            value: The metric to minimise (e.g. validation loss) at the
                current epoch.
            model: The model being trained; its weights are copied when a
                new best is reached and ``restore_best`` is ``True``.
            epoch: Current epoch index.

        Returns:
            ``True`` if training should stop, ``False`` otherwise. The base
            implementation always returns ``False``.
        """
        if value < self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return False

    def restore(self, model: Model) -> tuple[float | None, int | None]:
        """Restore the model to the best recorded state.

        Only has an effect when ``restore_best`` was set to ``True`` at
        construction and at least one call to ``step`` has been made.

        Args:
            model: The model whose weights will be replaced with the saved
                best-state snapshot.

        Returns:
            A ``(best_value, best_epoch)`` tuple if a best state was
            restored, otherwise ``(None, None)``.
        """
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
            return self.best_value, self.best_epoch

        return None, None
