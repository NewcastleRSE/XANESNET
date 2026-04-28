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

"""Class-level registry for XANESNET trainers."""

from collections.abc import Callable

from .base import Trainer


class TrainerRegistry:
    """Name-based registry for trainer classes."""

    _registry: dict[str, type[Trainer]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Trainer]], type[Trainer]]:
        """Register a trainer class under ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(ds_cls: type[Trainer]) -> type[Trainer]:
            """Register and return the decorated class unchanged."""
            if name in cls._registry:
                raise KeyError(f"Trainer '{name}' already registered")
            cls._registry[name] = ds_cls
            return ds_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Trainer]:
        """Return the trainer class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered trainer class.

        Raises:
            KeyError: If no trainer is registered under ``name``.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Trainer '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered trainer names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
