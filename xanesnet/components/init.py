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

"""Weight and bias initialisation registries for XANESNET model layers."""

from collections.abc import Callable
from typing import Protocol

import torch
from torch import nn


class WeightInitFn(Protocol):
    """Protocol for in-place weight initialisation callables."""

    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class WeightInitRegistry:
    """Class-level registry mapping weight initialisation names to initialiser callables."""

    _registry: dict[str, WeightInitFn] = {}

    @staticmethod
    def _noop(tensor: torch.Tensor, **_) -> torch.Tensor:
        """Return ``tensor`` unchanged (identity/default initialiser)."""
        return tensor

    @classmethod
    def register(cls, name: str) -> Callable[[WeightInitFn], WeightInitFn]:
        """Return a decorator that registers a weight initialiser under ``name``.

        Args:
            name: Unique lower-case identifier for the initialiser.

        Returns:
            A decorator that registers and returns the callable unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(fn: WeightInitFn) -> WeightInitFn:
            if name in cls._registry:
                raise KeyError(f"Weight initializer '{name}' already registered")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
        """Look up a weight initialiser and return a partially-applied callable.

        Args:
            name: Initialiser identifier (case-insensitive).
            **kwargs: Extra keyword arguments forwarded to the underlying initialiser.

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
            return fn(tensor, **kwargs)

        return wrapped

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered weight initialiser name strings.

        Returns:
            List of registered weight-initialiser identifiers.
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
    """Protocol for in-place bias initialisation callables."""

    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class BiasInitRegistry:
    """Class-level registry mapping bias initialisation names to initialiser callables."""

    _registry: dict[str, BiasInitFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[BiasInitFn], BiasInitFn]:
        """Return a decorator that registers a bias initialiser under ``name``.

        Args:
            name: Unique lower-case identifier for the initialiser.

        Returns:
            A decorator that registers and returns the callable unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(fn: BiasInitFn) -> BiasInitFn:
            if name in cls._registry:
                raise KeyError(f"Bias initializer '{name}' already registered")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> BiasInitFn:
        """Look up and return a registered bias initialiser callable.

        Args:
            name: Initialiser identifier (case-insensitive).

        Returns:
            The registered bias initialiser callable.

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Bias initializer '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered bias initialiser name strings.

        Returns:
            List of registered bias-initialiser identifiers.
        """
        return list(cls._registry.keys())


# register bias inits
BiasInitRegistry.register("zeros")(nn.init.zeros_)
BiasInitRegistry.register("ones")(nn.init.ones_)
