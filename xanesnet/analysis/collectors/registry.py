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

"""Registry for analysis collector implementations."""

from collections.abc import Callable

from .base import Collector


class CollectorRegistry:
    """Name-based registry for collector classes."""

    _registry: dict[str, type[Collector]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Collector]], type[Collector]]:
        """Register a collector class under ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(module_cls: type[Collector]) -> type[Collector]:
            """Register and return the decorated class unchanged."""
            if name in cls._registry:
                raise KeyError(f"Collector '{name}' already registered")
            cls._registry[name] = module_cls
            return module_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Collector]:
        """Return the collector class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered collector class.

        Raises:
            KeyError: If no collector is registered under ``name``.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Collector '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered collector names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
