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

"""Class-level registry for XANESNET datasource types."""

from collections.abc import Callable
from typing import TypeVar

from .base import DataSource

_T = TypeVar("_T", bound=DataSource)


class DataSourceRegistry:
    """Class-level registry mapping datasource type names to their ``DataSource`` classes."""

    _registry: dict[str, type[DataSource]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[_T]], type[_T]]:
        """Return a decorator that registers a datasource class under ``name``.

        Args:
            name: Unique lower-case identifier for the datasource type.

        Returns:
            A decorator that registers and returns the decorated class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(ds_cls: type[_T]) -> type[_T]:
            if name in cls._registry:
                raise KeyError(f"DataSource '{name}' already registered")
            cls._registry[name] = ds_cls
            return ds_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[DataSource]:
        """Look up and return a registered datasource class.

        Args:
            name: Datasource identifier (case-insensitive).

        Returns:
            The registered ``DataSource`` subclass.

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"DataSource '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered datasource name strings.

        Returns:
            List of registered datasource identifiers.
        """
        return list(cls._registry.keys())
