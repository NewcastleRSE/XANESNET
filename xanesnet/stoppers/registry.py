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

from .base import EarlyStopper


class EarlyStopperRegistry:
    _registry: dict[str, type[EarlyStopper]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[EarlyStopper]], type[EarlyStopper]]:
        name = name.lower()

        def decorator(stopper_cls: type[EarlyStopper]) -> type[EarlyStopper]:
            if name in cls._registry:
                raise KeyError(f"EarlyStopper '{name}' already registered")
            cls._registry[name] = stopper_cls
            return stopper_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[EarlyStopper]:
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"EarlyStopper '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())
