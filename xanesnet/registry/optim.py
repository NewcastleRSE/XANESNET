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

import torch.optim as optim

###############################################################################
################################## OPTIMIZER ##################################
###############################################################################


class OptimizerRegistry:
    _registry: dict[str, type[optim.Optimizer]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[optim.Optimizer]], type[optim.Optimizer]]:
        name = name.lower()

        def decorator(optim_cls: type[optim.Optimizer]) -> type[optim.Optimizer]:
            if name in cls._registry:
                raise KeyError(f"Optimizer '{name}' already registered")
            cls._registry[name] = optim_cls
            return optim_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[optim.Optimizer]:
        name = name.lower()
        if name not in cls._registry:
            raise KeyError(f"Optimizer '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())


# register optimizers
OptimizerRegistry.register("adam")(optim.Adam)
OptimizerRegistry.register("sgd")(optim.SGD)
OptimizerRegistry.register("rmsprop")(optim.RMSprop)
OptimizerRegistry.register("adamw")(optim.AdamW)
OptimizerRegistry.register("adagrad")(optim.Adagrad)
