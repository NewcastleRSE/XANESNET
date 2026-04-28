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

"""Activation function registry for XANESNET models."""

from collections.abc import Callable

import torch.nn as nn


class ActivationRegistry:
    """Class-level registry mapping activation function names to their ``nn.Module`` classes.

    All names are normalized to lower-case on registration and look-up.
    """

    _registry: dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
        """Return a decorator that registers an activation class under ``name``.

        Args:
            name: Unique lower-case identifier for the activation.

        Returns:
            A decorator that registers and returns the decorated class unchanged.

        Raises:
            KeyError: If ``name`` is already registered.
        """
        name = name.lower()

        def decorator(act_cls: type[nn.Module]) -> type[nn.Module]:
            """Register and return the decorated class unchanged."""
            if name in cls._registry:
                raise KeyError(f"Activation '{name}' already registered")
            cls._registry[name] = act_cls
            return act_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> nn.Module:
        """Instantiate and return a registered activation module.

        Args:
            name: Activation identifier (case-insensitive).
            **kwargs: Extra keyword arguments forwarded to the activation class constructor.

        Returns:
            An instantiated ``nn.Module`` for the requested activation.

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        name = name.lower()
        if name not in cls._registry:
            raise KeyError(f"Activation '{name}' not found in registry")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered activation name strings.

        Returns:
            List of registered activation identifiers.
        """
        return list(cls._registry.keys())


# register activations
ActivationRegistry.register("relu")(nn.ReLU)
ActivationRegistry.register("prelu")(nn.PReLU)
ActivationRegistry.register("tanh")(nn.Tanh)
ActivationRegistry.register("sigmoid")(nn.Sigmoid)
ActivationRegistry.register("elu")(nn.ELU)
ActivationRegistry.register("leakyrelu")(nn.LeakyReLU)
ActivationRegistry.register("selu")(nn.SELU)
ActivationRegistry.register("silu")(nn.SiLU)
ActivationRegistry.register("gelu")(nn.GELU)
