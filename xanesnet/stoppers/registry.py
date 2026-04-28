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
    """Class-level registry mapping string keys to ``EarlyStopper`` subclasses."""

    _registry: dict[str, type[EarlyStopper]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[EarlyStopper]], type[EarlyStopper]]:
        """Return a class decorator that registers an ``EarlyStopper`` under ``name``.

        Args:
            name: Registry key (lowercased automatically).

        Returns:
            A decorator that registers the decorated class and returns it
            unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(stopper_cls: type[EarlyStopper]) -> type[EarlyStopper]:
            if name in cls._registry:
                raise KeyError(f"EarlyStopper '{name}' already registered")
            cls._registry[name] = stopper_cls
            return stopper_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[EarlyStopper]:
        """Look up an ``EarlyStopper`` class by name.

        Args:
            name: Registry key (case-insensitive).

        Returns:
            The registered ``EarlyStopper`` subclass.

        Raises:
            KeyError: If ``name`` is not in the registry.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"EarlyStopper '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return the names of all registered early stoppers.

        Returns:
            List of registered registry keys.
        """
        return list(cls._registry.keys())
