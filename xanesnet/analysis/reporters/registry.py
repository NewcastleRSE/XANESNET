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

"""Registry for analysis reporter implementations."""

from collections.abc import Callable

from .base import Reporter


class ReporterRegistry:
    """Class-level mapping from reporter names to reporter classes."""

    _registry: dict[str, type[Reporter]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Reporter]], type[Reporter]]:
        """Create a decorator that registers a reporter class.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the reporter class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(reporter_cls: type[Reporter]) -> type[Reporter]:
            """Register ``reporter_cls`` under the normalized name.

            Args:
                reporter_cls: Reporter class to register.

            Returns:
                The unmodified reporter class.
            """
            if name in cls._registry:
                raise KeyError(f"Reporter '{name}' already registered")
            cls._registry[name] = reporter_cls
            return reporter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Reporter]:
        """Return the reporter class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered reporter class.

        Raises:
            KeyError: If no reporter is registered under ``name``.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Reporter '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered reporter names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
