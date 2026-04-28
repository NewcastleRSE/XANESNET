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

"""Weight and bias initialization registries for XANESNET model layers."""

from collections.abc import Callable
from typing import Protocol

import torch
from torch import nn


class WeightInitFn(Protocol):
    """Protocol for in-place weight initialization callables."""

    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Initialize ``tensor`` in place and return it."""
        ...


class WeightInitRegistry:
    """Class-level registry mapping weight initialization names to initializer callables."""

    _registry: dict[str, WeightInitFn] = {}

    @staticmethod
    def _noop(tensor: torch.Tensor, **_) -> torch.Tensor:
        """Return ``tensor`` unchanged (identity/default initializer)."""
        return tensor

    @classmethod
    def register(cls, name: str) -> Callable[[WeightInitFn], WeightInitFn]:
        """Return a decorator that registers a weight initializer under ``name``.

        Args:
            name: Unique lower-case identifier for the initializer.

        Returns:
            A decorator that registers and returns the callable unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(fn: WeightInitFn) -> WeightInitFn:
            """Register and return the decorated class unchanged."""
            if name in cls._registry:
                raise KeyError(f"Weight initializer '{name}' already registered")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
        """Look up a weight initializer and return a partially-applied callable.

        Args:
            name: Initializer identifier (case-insensitive).
            **kwargs: Extra keyword arguments forwarded to the underlying initializer.

        Returns:
            A single-argument callable ``(tensor) -> tensor`` with ``kwargs`` bound.

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Weight initializer '{name}' not found in registry")

        fn = cls._registry[name]

        def wrapped(tensor: torch.Tensor) -> torch.Tensor:
            """Apply the registered weight initializer."""
            return fn(tensor, **kwargs)

        return wrapped

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered weight initializer name strings.

        Returns:
            List of registered weight-initializer identifiers.
        """
        return list(cls._registry.keys())


# register weights inits
WeightInitRegistry.register("uniform")(nn.init.uniform_)
WeightInitRegistry.register("normal")(nn.init.normal_)
WeightInitRegistry.register("xavier_uniform")(nn.init.xavier_uniform_)
WeightInitRegistry.register("xavier_normal")(nn.init.xavier_normal_)
WeightInitRegistry.register("kaiming_uniform")(nn.init.kaiming_uniform_)
WeightInitRegistry.register("kaiming_normal")(nn.init.kaiming_normal_)
WeightInitRegistry.register("default")(WeightInitRegistry._noop)


class BiasInitFn(Protocol):
    """Protocol for in-place bias initialization callables."""

    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Initialize ``tensor`` in place and return it."""
        ...


class BiasInitRegistry:
    """Class-level registry mapping bias initialization names to initializer callables."""

    _registry: dict[str, BiasInitFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[BiasInitFn], BiasInitFn]:
        """Return a decorator that registers a bias initializer under ``name``.

        Args:
            name: Unique lower-case identifier for the initializer.

        Returns:
            A decorator that registers and returns the callable unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(fn: BiasInitFn) -> BiasInitFn:
            """Register and return the decorated class unchanged."""
            if name in cls._registry:
                raise KeyError(f"Bias initializer '{name}' already registered")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> BiasInitFn:
        """Look up and return a registered bias initializer callable.

        Args:
            name: Initializer identifier (case-insensitive).

        Returns:
            The registered bias initializer callable.

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Bias initializer '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered bias initializer name strings.

        Returns:
            List of registered bias-initializer identifiers.
        """
        return list(cls._registry.keys())


# register bias inits
BiasInitRegistry.register("zeros")(nn.init.zeros_)
BiasInitRegistry.register("ones")(nn.init.ones_)
