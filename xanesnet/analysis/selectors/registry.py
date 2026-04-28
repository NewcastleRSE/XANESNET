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

"""Registry for analysis selector implementations."""

from collections.abc import Callable

from .base import Selector


class SelectorRegistry:
    """Class-level mapping from selector names to selector classes."""

    _registry: dict[str, type[Selector]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Selector]], type[Selector]]:
        """Create a decorator that registers a selector class.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the selector class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(selector_cls: type[Selector]) -> type[Selector]:
            """Register ``selector_cls`` under the normalized name.

            Args:
                selector_cls: Selector class to register.

            Returns:
                The unmodified selector class.
            """
            if name in cls._registry:
                raise KeyError(f"Selector '{name}' already registered")
            cls._registry[name] = selector_cls
            return selector_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Selector]:
        """Return the selector class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered selector class.

        Raises:
            KeyError: If no selector is registered under ``name``.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Selector '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered selector names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
