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

DATASET_REGISTRY = {}
MODEL_REGISTRY = {}
DESCRIPTOR_REGISTRY = {}
SCHEME_REGISTRY = {}
LEARN_SCHEME_REGISTRY = {}
PREDICT_SCHEME_REGISTRY = {}

"""
This module provides central registries and decorators for registering
models, descriptors, and schemes. 

Usage:
    @register_model("mlp")
    @register_scheme("mlp", scheme_name="nn")
    class MLP(Model):
        ...

    @register_descriptor("wacsf")
    class WACSF:
        ...
        
    @register_dataset("xanesx")
    class XanesXDataset:
        ...

Once registered, the classes can be instantiated via corresponding
`create_*` factory functions using the registered name.
"""


def register_dataset(name):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def register_descriptor(name):
    def decorator(cls):
        DESCRIPTOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_scheme(model_name, scheme_name):
    def decorator(cls):
        if not SCHEME_REGISTRY:
            import xanesnet.scheme as scheme

            SCHEME_REGISTRY.update(
                {
                    "nn": {"learn": scheme.NNLearn, "predict": scheme.NNPredict},
                    "ae": {"learn": scheme.AELearn, "predict": scheme.AEPredict},
                    "mh": {"learn": scheme.MHLearn, "predict": scheme.MHPredict},
                    "ss": {"learn": scheme.SSLearn, "predict": scheme.SSPredict},
                    "aegan": {
                        "learn": scheme.AEGANLearn,
                        "predict": scheme.AEGANPredict,
                    },
                },
            )

        scheme = SCHEME_REGISTRY.get(scheme_name)
        if scheme is None:
            raise ValueError(f"Scheme '{scheme_name}' is not registered.")

        LEARN_SCHEME_REGISTRY[model_name] = scheme["learn"]
        PREDICT_SCHEME_REGISTRY[model_name] = scheme["predict"]
        return cls

    return decorator
