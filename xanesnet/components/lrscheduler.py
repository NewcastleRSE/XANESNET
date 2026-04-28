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

"""Learning rate scheduler registry for XANESNET training."""

from collections.abc import Callable

import torch.optim as optim


class LRSchedulerRegistry:
    """Class-level registry mapping scheduler names to their ``LRScheduler`` classes."""

    _registry: dict[str, type[optim.lr_scheduler.LRScheduler]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[optim.lr_scheduler.LRScheduler]], type[optim.lr_scheduler.LRScheduler]]:
        """Return a decorator that registers an LR scheduler class under ``name``.

        Args:
            name: Unique lower-case identifier for the scheduler.

        Returns:
            A decorator that registers and returns the decorated class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(lr_scheduler_cls: type[optim.lr_scheduler.LRScheduler]) -> type[optim.lr_scheduler.LRScheduler]:
            if name in cls._registry:
                raise KeyError(f"LRScheduler '{name}' already registered")
            cls._registry[name] = lr_scheduler_cls
            return lr_scheduler_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[optim.lr_scheduler.LRScheduler]:
        """Look up and return a registered LR scheduler class.

        Args:
            name: Scheduler identifier (case-insensitive).

        Returns:
            The registered ``LRScheduler`` class.

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        name = name.lower()
        if name not in cls._registry:
            raise KeyError(f"LRScheduler '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered scheduler name strings.

        Returns:
            List of registered learning-rate scheduler identifiers.
        """
        return list(cls._registry.keys())


class NoOpLRScheduler(optim.lr_scheduler.LRScheduler):
    """Learning rate scheduler that leaves all parameter group learning rates unchanged."""

    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        """Return the current learning rates unchanged."""
        return [group["lr"] for group in self.optimizer.param_groups]


# register lrschedulers
LRSchedulerRegistry.register("step")(optim.lr_scheduler.StepLR)
LRSchedulerRegistry.register("multistep")(optim.lr_scheduler.MultiStepLR)
LRSchedulerRegistry.register("exponential")(optim.lr_scheduler.ExponentialLR)
LRSchedulerRegistry.register("linear")(optim.lr_scheduler.LinearLR)
LRSchedulerRegistry.register("constant")(optim.lr_scheduler.ConstantLR)
LRSchedulerRegistry.register("none")(NoOpLRScheduler)
LRSchedulerRegistry.register("no")(NoOpLRScheduler)
