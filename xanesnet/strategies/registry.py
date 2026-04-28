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

"""Registry for XANESNET strategy implementations."""

from collections.abc import Callable

from .base import Strategy


class StrategyRegistry:
    """Class-level registry mapping string keys to ``Strategy`` subclasses."""

    _registry: dict[str, type[Strategy]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Strategy]], type[Strategy]]:
        """Return a class decorator that registers a ``Strategy`` under ``name``.

        Args:
            name: Registry key (lowercased automatically).

        Returns:
            A decorator that registers the decorated class and returns it
            unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(ds_cls: type[Strategy]) -> type[Strategy]:
            if name in cls._registry:
                raise KeyError(f"Strategy '{name}' already registered")
            cls._registry[name] = ds_cls
            return ds_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Strategy]:
        """Look up a ``Strategy`` class by name.

        Args:
            name: Registry key (case-insensitive).

        Returns:
            The registered ``Strategy`` subclass.

        Raises:
            KeyError: If ``name`` is not in the registry.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Strategy '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return the names of all registered strategies.

        Returns:
            List of registered registry keys.
        """
        return list(cls._registry.keys())
