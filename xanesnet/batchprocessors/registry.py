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


class BatchProcessorRegistry:
    _registry = {}

    @classmethod
    def register(cls, dataset_type: str, model_type: str):
        dataset_type = dataset_type.lower()
        model_type = model_type.lower()

        def decorator(adapter_cls):
            key = (dataset_type, model_type)
            if key in cls._registry:
                raise KeyError(f"BatchProcessor for {dataset_type}, {model_type} already registered")
            cls._registry[key] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, dataset_type: str, model_type: str):
        key = (dataset_type.lower(), model_type.lower())
        if key not in cls._registry:
            raise KeyError(f"No BatchProcessor registered for {dataset_type}, {model_type}")
        return cls._registry[key]()

    @classmethod
    def list(cls):
        return list(cls._registry.keys())
