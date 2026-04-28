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
    """Name-based registry for learning-rate scheduler classes."""

    _registry: dict[str, type[optim.lr_scheduler.LRScheduler]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[optim.lr_scheduler.LRScheduler]], type[optim.lr_scheduler.LRScheduler]]:
        """Register a learning-rate scheduler class under ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(lr_scheduler_cls: type[optim.lr_scheduler.LRScheduler]) -> type[optim.lr_scheduler.LRScheduler]:
            """Register and return the decorated class unchanged."""
            if name in cls._registry:
                raise KeyError(f"LRScheduler '{name}' already registered")
            cls._registry[name] = lr_scheduler_cls
            return lr_scheduler_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[optim.lr_scheduler.LRScheduler]:
        """Return the learning-rate scheduler class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered learning-rate scheduler class.

        Raises:
            KeyError: If no learning-rate scheduler is registered under ``name``.
        """
        name = name.lower()
        if name not in cls._registry:
            raise KeyError(f"LRScheduler '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered learning-rate scheduler names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())


class NoOpLRScheduler(optim.lr_scheduler.LRScheduler):
    """Learning rate scheduler that leaves all parameter group learning rates unchanged.

    Args:
        optimizer: Wrapped optimizer whose learning rates are reported unchanged.
        last_epoch: Last epoch index passed to :class:`torch.optim.lr_scheduler.LRScheduler`.
    """

    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        """Initialize ``NoOpLRScheduler``."""
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        """Return the current learning rates unchanged.

        Returns:
            Current learning rate from each optimizer parameter group.
        """
        return [group["lr"] for group in self.optimizer.param_groups]


# register lrschedulers
LRSchedulerRegistry.register("step")(optim.lr_scheduler.StepLR)
LRSchedulerRegistry.register("multistep")(optim.lr_scheduler.MultiStepLR)
LRSchedulerRegistry.register("exponential")(optim.lr_scheduler.ExponentialLR)
LRSchedulerRegistry.register("linear")(optim.lr_scheduler.LinearLR)
LRSchedulerRegistry.register("constant")(optim.lr_scheduler.ConstantLR)
LRSchedulerRegistry.register("none")(NoOpLRScheduler)
LRSchedulerRegistry.register("no")(NoOpLRScheduler)
