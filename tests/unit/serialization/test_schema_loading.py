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

"""Unit tests for schema loading"""

import copy
from typing import Any

import pytest
from jsonschema.validators import Draft202012Validator

from xanesnet.serialization import schema_validation as sv
from xanesnet.utils.exceptions import ConfigError

MODES = ["train", "infer", "analyze"]


def _find_keys(node: Any, key: str) -> list[Any]:
    """Collect every value stored under ``key`` anywhere inside a nested schema."""
    found: list[Any] = []
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                found.append(value)
            found.extend(_find_keys(value, key))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_keys(item, key))
    return found


###############################################################################
################################ ENTRY POINTS #################################
###############################################################################


@pytest.mark.parametrize("mode", MODES)
def test_schema_loads_as_a_valid_fully_resolved_draft_2020_12_schema(mode):
    schema = sv.load_config_schema(mode)
    Draft202012Validator.check_schema(schema)
    assert _find_keys(schema, "$ref") == [], "all $ref nodes must be inlined"


def test_loaded_schema_is_a_deep_copy_of_the_cache():
    first = sv.load_config_schema("train")
    pristine = copy.deepcopy(first)
    first["properties"].clear()
    assert sv.load_config_schema("train") == pristine


def test_unknown_mode_raises_and_lists_valid_modes():
    with pytest.raises(ConfigError, match="Unknown config validation mode.*analyze.*infer.*train"):
        sv.load_config_schema("nope")  # type: ignore[arg-type]


def test_train_schema_requires_the_expected_sections():
    required = set(sv.load_config_schema("train")["required"])
    assert required == {"datasource", "dataset", "model", "trainer", "strategy", "encodings"}
