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

from xanesnet.datasets import Dataset, DatasetRegistry
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.descriptors import Descriptor, DescriptorRegistry
from xanesnet.models import Model, ModelRegistry
from xanesnet.scheme import Learn, Predict, SchemeRegistry

###############################################################################
################################### FACTORY ###################################
###############################################################################

"""
Creates and returns instances of classes based on the given name.

Ensure that classes are registered using the appropriate decorators in their respective modules.
See 'xanesnet/registry' for more details.
"""


def create_datasource(name: str, **kwargs) -> DataSource:
    return DataSourceRegistry.get(name)(**kwargs)


def create_dataset(name: str, **kwargs) -> Dataset:
    return DatasetRegistry.get(name)(**kwargs)


def create_model(name: str, **kwargs) -> Model:
    return ModelRegistry.get(name)(**kwargs)


def create_descriptor(name: str, **kwargs) -> Descriptor:
    return DescriptorRegistry.get(name)(**kwargs)


def create_descriptors(config: Dict = None) -> List:
    descriptor_list = []

    for descriptor in config:
        params = descriptor.get("params", {})
        descriptor = create_descriptor(descriptor["type"], **params)
        descriptor_list.append(descriptor)

    return descriptor_list


def create_learn_scheme(name: str, model: Model, dataset: Dataset, **kwargs) -> Learn:
    return SchemeRegistry.get_learn(name)(model, dataset, **kwargs)


def create_predict_scheme(name: str, dataset: Dataset, **kwargs) -> Predict:
    return SchemeRegistry.get_predict(name)(dataset, **kwargs)
