# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either Version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

"""Registry for dataset implementations."""

from collections.abc import Callable
from typing import TypeVar

from .base import Dataset

_DatasetT = TypeVar("_DatasetT", bound=type[Dataset])


class DatasetRegistry:
    """Name-based registry for dataset classes."""

    _registry: dict[str, type[Dataset]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[_DatasetT], _DatasetT]:
        """Register a dataset class under ``name``.

        Args:
            name: Registry key used in configuration files.

        Returns:
            Decorator that registers and returns the dataset class unchanged.
        """
        name = name.lower()

        def decorator(ds_cls: _DatasetT) -> _DatasetT:
            """Register and return ``ds_cls`` unchanged."""
            if name in cls._registry:
                raise KeyError(f"Dataset '{name}' already registered")
            cls._registry[name] = ds_cls
            return ds_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Dataset]:
        """Return the dataset class registered as ``name``.

        Args:
            name: Registry key to resolve.

        Returns:
            Registered dataset class.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Dataset '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered dataset names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
