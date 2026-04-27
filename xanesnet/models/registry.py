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
    """Class-level registry mapping model name strings to their implementation classes.

    All names are normalised to lower-case on registration and look-up.
    """

    _registry: dict[str, type[Model]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[ModelType]], type[ModelType]]:
        """Return a decorator that registers a model class under ``name``.

        Args:
            name: Unique lower-case identifier for the model.

        Returns:
            A decorator that registers and returns the decorated class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(ds_cls: type[ModelType]) -> type[ModelType]:
            if name in cls._registry:
                raise KeyError(f"Model '{name}' already registered")
            cls._registry[name] = ds_cls
            return ds_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Model]:
        """Look up and return a registered model class by name.

        Args:
            name: Model identifier (case-insensitive).

        Returns:
            The registered model class.

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered model name strings.

        Returns:
            Sorted list of registered model names.
        """
        return list(cls._registry.keys())
