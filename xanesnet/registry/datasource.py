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


class DataSourceRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(ds_cls):
            if name in cls._registry:
                raise KeyError(f"DataSource '{name}' already registered")
            cls._registry[name] = ds_cls
            return ds_cls

        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise KeyError(f"DataSource '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list(cls):
        return list(cls._registry.keys())
