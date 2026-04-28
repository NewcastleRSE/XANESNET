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

"""Registry for analysis plotter implementations."""

from collections.abc import Callable

from .base import Plotter


class PlotterRegistry:
    """Class-level mapping from plotter names to plotter classes."""

    _registry: dict[str, type[Plotter]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Plotter]], type[Plotter]]:
        """Create a decorator that registers a plotter class.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the plotter class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(plotter_cls: type[Plotter]) -> type[Plotter]:
            """Register ``plotter_cls`` under the normalized name.

            Args:
                plotter_cls: Plotter class to register.

            Returns:
                The unmodified plotter class.
            """
            if name in cls._registry:
                raise KeyError(f"Plotter '{name}' already registered")
            cls._registry[name] = plotter_cls
            return plotter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Plotter]:
        """Return the plotter class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered plotter class.

        Raises:
            KeyError: If no plotter is registered under ``name``.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Plotter '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered plotter names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
