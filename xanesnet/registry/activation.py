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

from typing import Callable

import torch.nn as nn


class ActivationRegistry:
    _registry: dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
        name = name.lower()

        def decorator(act_cls: type[nn.Module]) -> type[nn.Module]:
            if name in cls._registry:
                raise KeyError(f"Activation '{name}' already registered")
            cls._registry[name] = act_cls
            return act_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> nn.Module:
        name = name.lower()
        if name not in cls._registry:
            raise KeyError(f"Activation '{name}' not found in registry")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
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
