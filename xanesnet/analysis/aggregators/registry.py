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

"""Registry for analysis aggregator implementations."""

from collections.abc import Callable

from .base import Aggregator


class AggregatorRegistry:
    """Class-level mapping from aggregator names to aggregator classes."""

    _registry: dict[str, type[Aggregator]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Aggregator]], type[Aggregator]]:
        """Create a decorator that registers an aggregator class.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the aggregator class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(aggregator_cls: type[Aggregator]) -> type[Aggregator]:
            """Register ``aggregator_cls`` under the normalized name.

            Args:
                aggregator_cls: Aggregator class to register.

            Returns:
                The unmodified aggregator class.
            """
            if name in cls._registry:
                raise KeyError(f"Aggregator '{name}' already registered")
            cls._registry[name] = aggregator_cls
            return aggregator_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Aggregator]:
        """Return the aggregator class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered aggregator class.

        Raises:
            KeyError: If no aggregator is registered under ``name``.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Aggregator '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered aggregator names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
