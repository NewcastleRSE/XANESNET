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

"""Unit tests for the calling contract of ``validate_config_schema`` (mutation, return value, error format)."""

import copy

import pytest

from xanesnet.serialization.schema_validation import validate_config_schema
from xanesnet.utils.exceptions import ConfigError


def test_valid_config_is_updated_in_place_and_returned(train_config):
    returned = validate_config_schema(train_config, "train")

    assert train_config == returned
    assert train_config["device"] == "cpu", "defaults must be written back into the caller's dictionary"
    assert returned is not train_config


def test_in_place_update_keeps_the_same_dictionary_object(train_config):
    identity = id(train_config)
    validate_config_schema(train_config, "train")
    assert id(train_config) == identity


def test_returned_config_is_independent_of_the_input(train_config):
    returned = validate_config_schema(train_config, "train")
    returned["model"]["hidden_size"] = 999
    assert train_config["model"]["hidden_size"] == 8


def test_invalid_config_is_left_untouched(train_config):
    train_config["model"]["model_type"] = "nope"
    snapshot = copy.deepcopy(train_config)

    with pytest.raises(ConfigError):
        validate_config_schema(train_config, "train")

    assert train_config == snapshot, "failed validation must not leak defaults into the caller's config"


def test_validation_is_idempotent(train_config):
    once = validate_config_schema(train_config, "train")
    twice = validate_config_schema(copy.deepcopy(once), "train")
    assert twice == once


def test_the_shallowest_problem_is_reported_first(train_config):
    train_config["model"]["hidden_size"] = 0  # config.model.* (depth 2)
    train_config["unexpected"] = 1  # config (depth 0)
    with pytest.raises(ConfigError, match=r"^Invalid train config at config:"):
        validate_config_schema(train_config, "train")


def test_error_for_a_list_item_uses_an_indexed_path(train_config):
    train_config["encodings"] = [{"encoding_type": "identity"}, {"encoding_type": "nope"}]
    with pytest.raises(ConfigError, match=r"config\.encodings\[1\]"):
        validate_config_schema(train_config, "train")


def test_union_errors_list_details_from_the_candidate_branches(train_config):
    train_config["model"]["hidden_size"] = 0
    with pytest.raises(ConfigError) as excinfo:
        validate_config_schema(train_config, "train")
    assert "Details:" in str(excinfo.value)


def test_error_messages_are_single_line_and_bounded(train_config):
    train_config["model"]["model_type"] = "nope"
    with pytest.raises(ConfigError) as excinfo:
        validate_config_schema(train_config, "train")
    message = str(excinfo.value)
    assert "\n" not in message
    assert len(message) < 2000


def test_unknown_mode_is_rejected(train_config):
    with pytest.raises(ConfigError, match="Unknown config validation mode"):
        validate_config_schema(train_config, "bogus")  # type: ignore[arg-type]
