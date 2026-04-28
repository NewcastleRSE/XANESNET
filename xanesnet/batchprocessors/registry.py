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
    """Registry for batch processor classes keyed by dataset and model type."""

    _registry: dict[tuple[str, str], type[BatchProcessor]] = {}

    @classmethod
    def register(cls, dataset_type: str, model_type: str) -> Callable[[type[BatchProcessor]], type[BatchProcessor]]:
        """Register a batch processor class for a dataset/model pair.

        Args:
            dataset_type: Dataset registry key. Matching is case-insensitive.
            model_type: Model registry key. Matching is case-insensitive.

        Returns:
            Decorator that registers and returns the class unchanged.

        Raises:
            KeyError: If the dataset/model pair is already registered.
        """
        dataset_type = dataset_type.lower()
        model_type = model_type.lower()

        def decorator(adapter_cls: type[BatchProcessor]) -> type[BatchProcessor]:
            """Register and return the decorated class unchanged."""
            key = (dataset_type, model_type)
            if key in cls._registry:
                raise KeyError(f"BatchProcessor for {dataset_type}, {model_type} already registered")
            cls._registry[key] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, dataset_type: str, model_type: str) -> type[BatchProcessor]:
        """Return the batch processor class registered for a dataset/model pair.

        Args:
            dataset_type: Dataset registry key. Matching is case-insensitive.
            model_type: Model registry key. Matching is case-insensitive.

        Returns:
            Registered batch processor class.

        Raises:
            KeyError: If no processor is registered for the pair.
        """
        key = (dataset_type.lower(), model_type.lower())
        if key not in cls._registry:
            raise KeyError(f"No BatchProcessor registered for {dataset_type}, {model_type}")
        return cls._registry[key]

    @classmethod
    def list(cls) -> list[tuple[str, str]]:
        """Return all registered dataset/model key pairs.

        Returns:
            Registry keys in insertion order.
        """
        return list(cls._registry.keys())
