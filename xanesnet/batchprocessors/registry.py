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

"""Registry for XANESNET batch processor classes."""

from collections.abc import Callable

from .base import BatchProcessor


class BatchProcessorRegistry:
    """Class-level registry mapping (dataset_type, model_type) pairs to batch processor classes.

    All name strings are normalised to lower-case on registration and look-up.
    """

    _registry: dict[tuple[str, str], type[BatchProcessor]] = {}

    @classmethod
    def register(cls, dataset_type: str, model_type: str) -> Callable[[type[BatchProcessor]], type[BatchProcessor]]:
        """Return a decorator that registers a batch processor class.

        Args:
            dataset_type: Dataset identifier (case-insensitive).
            model_type: Model identifier (case-insensitive).

        Returns:
            A decorator that registers and returns the decorated class unchanged.

        Raises:
            KeyError: If (dataset_type, model_type) is already registered.
        """
        dataset_type = dataset_type.lower()
        model_type = model_type.lower()

        def decorator(adapter_cls: type[BatchProcessor]) -> type[BatchProcessor]:
            key = (dataset_type, model_type)
            if key in cls._registry:
                raise KeyError(f"BatchProcessor for {dataset_type}, {model_type} already registered")
            cls._registry[key] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, dataset_type: str, model_type: str) -> type[BatchProcessor]:
        """Look up and return a registered batch processor class.

        Args:
            dataset_type: Dataset identifier (case-insensitive).
            model_type: Model identifier (case-insensitive).

        Returns:
            The registered batch processor class.

        Raises:
            KeyError: If no processor is registered for the given (dataset_type, model_type) pair.
        """
        key = (dataset_type.lower(), model_type.lower())
        if key not in cls._registry:
            raise KeyError(f"No BatchProcessor registered for {dataset_type}, {model_type}")
        return cls._registry[key]

    @classmethod
    def list(cls) -> list[tuple[str, str]]:
        """Return all registered (dataset_type, model_type) key pairs.

        Returns:
            List of registered (dataset_type, model_type) tuples.
        """
        return list(cls._registry.keys())
