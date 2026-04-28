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

"""Registry for XANESNET early-stopper implementations."""

from collections.abc import Callable

from .base import EarlyStopper


class EarlyStopperRegistry:
    """Name-based registry for early stopper classes."""

    _registry: dict[str, type[EarlyStopper]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[EarlyStopper]], type[EarlyStopper]]:
        """Register an early stopper class under ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(stopper_cls: type[EarlyStopper]) -> type[EarlyStopper]:
            """Register and return the decorated class unchanged."""
            if name in cls._registry:
                raise KeyError(f"EarlyStopper '{name}' already registered")
            cls._registry[name] = stopper_cls
            return stopper_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[EarlyStopper]:
        """Return the early stopper class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered early stopper class.

        Raises:
            KeyError: If no early stopper is registered under ``name``.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"EarlyStopper '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered early stopper names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
