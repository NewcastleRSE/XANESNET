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

"""Class-level registry for XANESNET inferencers."""

from collections.abc import Callable

from .base import Inferencer


class InferencerRegistry:
    """Name-based registry for inferencer classes."""

    _registry: dict[str, type[Inferencer]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Inferencer]], type[Inferencer]]:
        """Register an inferencer class under ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(ds_cls: type[Inferencer]) -> type[Inferencer]:
            """Register and return the decorated class unchanged."""
            if name in cls._registry:
                raise KeyError(f"Inferencer '{name}' already registered")
            cls._registry[name] = ds_cls
            return ds_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Inferencer]:
        """Return the inferencer class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered inferencer class.

        Raises:
            KeyError: If no inferencer is registered under ``name``.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Inferencer '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered inferencer names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
