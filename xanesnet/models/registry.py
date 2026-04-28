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

"""Registry for XANESNET model classes."""

from collections.abc import Callable
from typing import TypeVar

from .base import Model

ModelType = TypeVar("ModelType", bound=Model)


class ModelRegistry:
    """Name-based registry for model classes."""

    _registry: dict[str, type[Model]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[ModelType]], type[ModelType]]:
        """Register a model class under ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(ds_cls: type[ModelType]) -> type[ModelType]:
            """Register and return the decorated class unchanged."""
            if name in cls._registry:
                raise KeyError(f"Model '{name}' already registered")
            cls._registry[name] = ds_cls
            return ds_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Model]:
        """Return the model class registered as ``name``.

        Args:
            name: Registry key. Matching is case-insensitive.

        Returns:
            Registered model class.

        Raises:
            KeyError: If no model is registered under ``name``.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered model names.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
