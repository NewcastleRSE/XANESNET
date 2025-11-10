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
from typing import Dict, List

from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.descriptors.base_descriptor import BaseDescriptor
from xanesnet.models.base_model import Model
from xanesnet.registry import (
    MODEL_REGISTRY,
    LEARN_SCHEME_REGISTRY,
    PREDICT_SCHEME_REGISTRY,
    DESCRIPTOR_REGISTRY,
    DATASET_REGISTRY,
)
from xanesnet.scheme import Learn, Predict
from xanesnet.utils.mode import Mode

"""
Factory functions to create instance of model, descriptor or scheme
"""


def create_dataset(name: str, **kwargs) -> BaseDataset:
    """
    Create and return an instance of a dataset class based on the given name.

    Dataset must be registered using the @register_dataset("dataset_name") decorator.
    See `dataset/xanesx.py` for an example of how to register a dataset class.

    Args:
        name (str): The name of the dataset to create.
        **kwargs: Additional keyword arguments passed to the dataset constructor.

    Returns:
        An instance of the dataset.

    Raises:
        ValueError: If the specified dataset name is not registered.
    """
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name](**kwargs)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")


def create_model(name: str, **kwargs) -> Model:
    """
    Create and return an instance of a model class based on the given name.

    Models must be registered using the @register_model("model_name") decorator.
    See `models/mlp.py` for an example of how to register a model class.

    Args:
        name (str): The name of the model to create.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        An instance of the model.

    Raises:
        ValueError: If the specified model name is not registered.
    """
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name](**kwargs)
    else:
        raise ValueError(f"Unsupported model name: {name}")


def create_descriptor(name: str, **kwargs) -> BaseDescriptor:
    """
    Create and return an instance of a descriptor class based on the given name.

    Descriptors must be registered using the
    @register_descriptor("descriptor_name") decorator.
    See `descriptor/wacsf.py` for an example of how to register a descriptor class.

    Args:
        name (str): The name of the descriptor to create.
        **kwargs: Additional keyword arguments passed to the descriptor constructor.

    Returns:
        An instance of the descriptor.

    Raises:
        ValueError: If the specified descriptor name is not registered.
    """
    if name in DESCRIPTOR_REGISTRY:
        return DESCRIPTOR_REGISTRY[name](**kwargs)
    else:
        raise ValueError(f"Unsupported descriptor name: {name}")


def create_descriptors(config: Dict = None) -> List:
    """
    Create and return a list of descriptor instances.
    """
    descriptor_list = []

    for descriptor in config:
        params = descriptor.get("params", {})
        descriptor = create_descriptor(descriptor["type"], **params)
        descriptor_list.append(descriptor)

    return descriptor_list


def create_learn_scheme(
    name: str, model: Model, dataset: BaseDataset, **kwargs
) -> Learn:
    """
    Create and return an instance of a learning scheme based on the given scheme name.

    Learning schemes define how a model is trained and are registered using the
    @register_scheme(model_name, scheme_name) decorator in MODEL classes.
    See `models/mlp.py` for an example.

    Args:
        name (str): The name of the scheme.
        model: The model instance to be trained.
        dataset: The dataset instance used for training.
        **kwargs: Additional keyword arguments passed to the learning scheme constructor.

    Returns:
        An instance of the learning scheme associated with the given model.

    Raises:
        ValueError: If the specified learning scheme name is not registered.
    """
    if name in LEARN_SCHEME_REGISTRY:
        return LEARN_SCHEME_REGISTRY[name](model, dataset, **kwargs)
    raise ValueError(f"Unsupported learn scheme for the model: {name}")


def create_predict_scheme(name: str, dataset: BaseDataset, **kwargs) -> Predict:
    """
    Create and return an instance of a prediction scheme based on the given scheme name.

    Prediction schemes are registered using the
    @register_scheme(model_name, scheme_name) decorator in MODEL classes.
    See `models/mlp.py` for an example.

    Args:
        name (str): The name of the scheme.
        dataset: The dataset instance used for prediction.
        **kwargs: Additional keyword arguments passed to the prediction scheme constructor.

    Returns:
        An instance of the prediction scheme associated with the given model.

    Raises:
        ValueError: If the specified prediction scheme name is not registered.
    """
    if name in PREDICT_SCHEME_REGISTRY:
        return PREDICT_SCHEME_REGISTRY[name](dataset, **kwargs)
    else:
        raise ValueError(f"Unsupported predict scheme for the model: {name}")
