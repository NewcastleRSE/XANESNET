# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""The packaged JSON Schemas and the Python registries must advertise exactly the same extension points.

A type that is registered but missing from the schema cannot be selected from a config file; a type that is in the
schema but not registered passes validation and then fails at run time. Both directions are checked here.
"""

from importlib import resources

import pytest
import yaml

from xanesnet.components import (
    ActivationRegistry,
    BiasInitRegistry,
    LRSchedulerRegistry,
    OptimizerRegistry,
    WeightInitRegistry,
)
from xanesnet.datasets import DatasetRegistry
from xanesnet.datasources import DataSourceRegistry
from xanesnet.descriptors import DescriptorRegistry
from xanesnet.encodings import SpectraEncodingRegistry
from xanesnet.losses import LossRegistry
from xanesnet.models import ModelRegistry
from xanesnet.regularizers import RegularizerRegistry
from xanesnet.runners.inferencers import InferencerRegistry
from xanesnet.runners.trainers import TrainerRegistry
from xanesnet.stoppers import EarlyStopperRegistry
from xanesnet.strategies import StrategyRegistry

# Other unit tests register fakes under this prefix; they are not part of the public surface.
TEST_DOUBLE_PREFIX = "unit_test_"

# (schema file, definition name, registry)
SYNCHRONISED = {
    "activation": ("components/component_types.schema.yaml", "activationName", ActivationRegistry),
    "optimizer": ("components/component_types.schema.yaml", "optimizerName", OptimizerRegistry),
    "weight_init": ("components/component_types.schema.yaml", "weightInitName", WeightInitRegistry),
    "bias_init": ("components/component_types.schema.yaml", "biasInitName", BiasInitRegistry),
    "lr_scheduler": ("components/component_types.schema.yaml", "lrSchedulerType", LRSchedulerRegistry),
    "dataset": ("datasets/dataset_types.schema.yaml", "datasetType", DatasetRegistry),
    "datasource": ("datasources/datasource_types.schema.yaml", "datasourceType", DataSourceRegistry),
    "descriptor": ("descriptors/descriptor_types.schema.yaml", "descriptorType", DescriptorRegistry),
    "encoding": ("encodings/encoding_types.schema.yaml", "encodingType", SpectraEncodingRegistry),
    "loss": ("losses/loss_types.schema.yaml", "lossType", LossRegistry),
    "model": ("models/model_types.schema.yaml", "modelType", ModelRegistry),
    "regularizer": ("regularizers/regularizer_types.schema.yaml", "regularizerType", RegularizerRegistry),
    "trainer": ("runners/runner_types.schema.yaml", "trainerType", TrainerRegistry),
    "inferencer": ("runners/runner_types.schema.yaml", "inferencerType", InferencerRegistry),
    "early_stopper": ("stoppers/early_stopper_types.schema.yaml", "earlyStopperType", EarlyStopperRegistry),
    "strategy": ("strategies/strategy_types.schema.yaml", "strategyType", StrategyRegistry),
}


def _schema_enum(schema_file: str, definition: str) -> list[str]:
    path = resources.files("xanesnet") / "schemas" / schema_file
    return yaml.safe_load(path.read_text(encoding="utf-8"))["$defs"][definition]["enum"]


@pytest.mark.parametrize("name", SYNCHRONISED)
def test_schema_enum_matches_registry_keys(name):
    schema_file, definition, registry = SYNCHRONISED[name]

    # Registry lookups are case-insensitive, and the schema may list both spellings (e.g. ``adam`` and ``Adam``).
    in_schema = {value.lower() for value in _schema_enum(schema_file, definition)}
    in_registry = {key for key in registry.list() if not key.startswith(TEST_DOUBLE_PREFIX)}

    assert in_schema - in_registry == set(), f"in the schema but not registered: {sorted(in_schema - in_registry)}"
    assert in_registry - in_schema == set(), f"registered but not in the schema: {sorted(in_registry - in_schema)}"


@pytest.mark.parametrize("name", SYNCHRONISED)
def test_every_schema_entry_resolves_through_the_registry(name):
    schema_file, definition, registry = SYNCHRONISED[name]
    for value in _schema_enum(schema_file, definition):
        registry.get(value)  # raises KeyError for unregistered entries


@pytest.mark.parametrize("name", SYNCHRONISED)
def test_registries_are_never_empty(name):
    assert SYNCHRONISED[name][2].list()
