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

from .base import PerSampleModule


class PerSampleRegistry:
    _registry: dict[str, type[PerSampleModule]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[PerSampleModule]], type[PerSampleModule]]:
        name = name.lower()

        def decorator(module_cls: type[PerSampleModule]) -> type[PerSampleModule]:
            if name in cls._registry:
                raise KeyError(f"PerSampleModule '{name}' already registered")
            cls._registry[name] = module_cls
            return module_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[PerSampleModule]:
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"PerSampleModule '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())
