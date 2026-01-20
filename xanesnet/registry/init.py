"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import nn

###############################################################################
################################### WEIGHTS ###################################
###############################################################################


class WeightInitFn(Protocol):
    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class WeightInitRegistry:
    _registry: dict[str, WeightInitFn] = {}

    @staticmethod
    def _noop(tensor: torch.Tensor, **_) -> torch.Tensor:
        return tensor

    @classmethod
    def register(cls, name: str) -> Callable[[WeightInitFn], WeightInitFn]:
        name = name.lower()

        def decorator(fn: WeightInitFn) -> WeightInitFn:
            if name in cls._registry:
                raise KeyError(f"Weight initializer '{name}' already registered")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Weight initializer '{name}' not found in registry")

        fn = cls._registry[name]

        def wrapped(tensor: torch.Tensor) -> torch.Tensor:
            return fn(tensor, **kwargs)

        return wrapped

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())


# register weights inits
WeightInitRegistry.register("uniform")(nn.init.uniform_)
WeightInitRegistry.register("normal")(nn.init.normal_)
WeightInitRegistry.register("xavier_uniform")(nn.init.xavier_uniform_)
WeightInitRegistry.register("xavier_normal")(nn.init.xavier_normal_)
WeightInitRegistry.register("kaiming_uniform")(nn.init.kaiming_uniform_)
WeightInitRegistry.register("kaiming_normal")(nn.init.kaiming_normal_)
WeightInitRegistry.register("default")(WeightInitRegistry._noop)

###############################################################################
####################################  BIAS ####################################
###############################################################################


class BiasInitFn(Protocol):
    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class BiasInitRegistry:
    _registry: dict[str, BiasInitFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[BiasInitFn], BiasInitFn]:
        name = name.lower()

        def decorator(fn: BiasInitFn) -> BiasInitFn:
            if name in cls._registry:
                raise KeyError(f"Bias initializer '{name}' already registered")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> BiasInitFn:
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Bias initializer '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())


# register bias inits
BiasInitRegistry.register("zeros")(nn.init.zeros_)
BiasInitRegistry.register("ones")(nn.init.ones_)
