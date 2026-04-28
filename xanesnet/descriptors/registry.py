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

"""Class-level registry for XANESNET descriptor types."""

from collections.abc import Callable
from typing import TypeVar

from .base import Descriptor

_T = TypeVar("_T", bound=Descriptor)


class DescriptorRegistry:
    """Class-level registry mapping descriptor names to their ``Descriptor`` subclasses."""

    _registry: dict[str, type[Descriptor]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[_T]], type[_T]]:
        """Return a decorator that registers a descriptor class under ``name``.

        Args:
            name: Unique lower-case identifier for the descriptor type.

        Returns:
            A decorator that registers and returns the decorated class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(ds_cls: type[_T]) -> type[_T]:
            if name in cls._registry:
                raise KeyError(f"Descriptor '{name}' already registered")
            cls._registry[name] = ds_cls
            return ds_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Descriptor]:
        """Look up and return a registered descriptor class.

        Args:
            name: Descriptor identifier (case-insensitive).

        Returns:
            The registered ``Descriptor`` subclass.

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Descriptor '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered descriptor name strings.

        Returns:
            List of registered descriptor identifiers.
        """
        return list(cls._registry.keys())
