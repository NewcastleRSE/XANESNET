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

from typing import Dict, Optional, Type


class SchemeRegistry:
    _schemes: Dict[str, Dict[str, Optional[Type]]] = {}
    _learn_registry: Dict[str, Type] = {}
    _predict_registry: Dict[str, Optional[Type]] = {}
    _initialized = False

    @classmethod
    def _initialize_schemes(cls):
        if cls._initialized:
            return
        # lazy import to avoid circular imports
        import xanesnet.scheme as scheme

        cls._schemes.update(
            {
                "nn": {"learn": scheme.NNLearn, "predict": scheme.NNPredict},
                "ss": {"learn": scheme.SSLearn, "predict": scheme.SSPredict},
            }
        )
        cls._initialized = True

    @classmethod
    def register(cls, model_name: str, scheme_name: str):
        """
        Decorator to register a learn/predict scheme for a model.
        """

        def decorator(model_cls: Type):
            cls._initialize_schemes()
            scheme = cls._schemes.get(scheme_name)
            if scheme is None:
                raise ValueError(f"Scheme '{scheme_name}' is not registered.")

            cls._learn_registry[model_name] = scheme["learn"]
            cls._predict_registry[model_name] = scheme["predict"]
            return model_cls

        return decorator

    @classmethod
    def get_learn(cls, model_name: str) -> Type:
        if model_name not in cls._learn_registry:
            raise ValueError(f"No learn scheme registered for model: {model_name}")
        return cls._learn_registry[model_name]

    @classmethod
    def get_predict(cls, model_name: str) -> Optional[Type]:
        if model_name not in cls._predict_registry:
            raise ValueError(f"No predict scheme registered for model: {model_name}")
        return cls._predict_registry[model_name]

    @classmethod
    def list_models(cls):
        return list(cls._learn_registry.keys())

    @classmethod
    def list_schemes(cls):
        cls._initialize_schemes()
        return list(cls._schemes.keys())
