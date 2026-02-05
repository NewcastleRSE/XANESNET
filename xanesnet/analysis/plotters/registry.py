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

from .base import Plotter


class PlotterRegistry:
    _registry: dict[str, type[Plotter]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Plotter]], type[Plotter]]:
        name = name.lower()

        def decorator(plotter_cls: type[Plotter]) -> type[Plotter]:
            if name in cls._registry:
                raise KeyError(f"Plotter '{name}' already registered")
            cls._registry[name] = plotter_cls
            return plotter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Plotter]:
        name = name.lower()

        if name not in cls._registry:
            raise KeyError(f"Plotter '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())
