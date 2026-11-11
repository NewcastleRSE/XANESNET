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

"""Unit tests for the typed ``Config`` wrapper: accessors, conversion and merging."""

import pytest
import yaml

from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError


@pytest.fixture
def config() -> Config:
    return Config(
        {
            "name": "run",
            "count": 3,
            "rate": 0.5,
            "flag": True,
            "nothing": None,
            "items": [{"a": 1}, {"b": 2}],
            "numbers": [1, 2, 3],
            "model": {"model_type": "mlp", "layers": [{"size": 4}], "dropout": 0.1},
        }
    )


###############################################################################
############################### TYPED ACCESSORS ###############################
###############################################################################

TYPED_CASES = [
    ("get_str", "name", "run"),
    ("get_int", "count", 3),
    ("get_float", "rate", 0.5),
    ("get_bool", "flag", True),
]


@pytest.mark.parametrize("getter, key, expected", TYPED_CASES)
def test_required_accessor_returns_the_value(config, getter, key, expected):
    value = getattr(config, getter)(key)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize("getter, key, expected", TYPED_CASES)
def test_optional_accessor_returns_the_value_when_present(config, getter, key, expected):
    assert getattr(config, getter.replace("get_", "get_optional_"))(key) == expected


@pytest.mark.parametrize("getter", ["get_str", "get_int", "get_float", "get_bool"])
@pytest.mark.parametrize("key", ["absent", "nothing"], ids=["missing", "null"])
def test_required_accessor_treats_missing_and_null_as_missing(config, getter, key):
    with pytest.raises(ConfigError, match=f"Key '{key}' is missing"):
        getattr(config, getter)(key)


@pytest.mark.parametrize("getter", ["get_str", "get_int", "get_float", "get_bool"])
@pytest.mark.parametrize("key", ["absent", "nothing"], ids=["missing", "null"])
def test_optional_accessor_returns_none_for_missing_and_null(config, getter, key):
    assert getattr(config, getter.replace("get_", "get_optional_"))(key) is None


@pytest.mark.parametrize(
    "getter, key",
    [
        ("get_str", "count"),
        ("get_int", "name"),
        ("get_int", "rate"),
        ("get_float", "count"),
        ("get_float", "name"),
        ("get_bool", "count"),
        ("get_bool", "name"),
    ],
)
def test_wrong_type_is_rejected(config, getter, key):
    with pytest.raises(ConfigError, match=f"Key '{key}' is not of type"):
        getattr(config, getter)(key)


@pytest.mark.parametrize("getter, key", [("get_optional_int", "name"), ("get_optional_str", "count")])
def test_optional_accessor_still_validates_the_type_of_present_values(config, getter, key):
    with pytest.raises(ConfigError, match="is not of type"):
        getattr(config, getter)(key)


def test_raw_get_returns_any_value_including_none(config):
    assert config.get("count") == 3
    assert config.get("nothing") is None


def test_raw_get_of_absent_key_raises(config):
    with pytest.raises(ConfigError, match="Key 'absent' is missing"):
        config.get("absent")


###############################################################################
################################## SECTIONS ###################################
###############################################################################


def test_section_returns_a_config_over_the_nested_dict(config):
    section = config.section("model")
    assert isinstance(section, Config)
    assert section.get_str("model_type") == "mlp"


@pytest.mark.parametrize("key", ["absent", "nothing"])
def test_missing_section_raises(config, key):
    with pytest.raises(ConfigError, match="is missing"):
        config.section(key)


def test_non_dict_section_raises(config):
    with pytest.raises(ConfigError, match="is not of type dict"):
        config.section("numbers")


def test_optional_section(config):
    assert config.optional_section("absent") is None
    assert config.optional_section("nothing") is None
    assert config.optional_section("model").get_float("dropout") == 0.1  # type: ignore[union-attr]
    with pytest.raises(ConfigError, match="is not of type dict"):
        config.optional_section("count")


def test_get_config_list_wraps_each_dictionary(config):
    items = config.get_config_list("items")
    assert [type(item) for item in items] == [Config, Config]
    assert [item.as_dict() for item in items] == [{"a": 1}, {"b": 2}]


def test_get_config_list_rejects_non_dictionary_elements(config):
    with pytest.raises(ConfigError, match="must be a list of dictionaries"):
        config.get_config_list("numbers")


def test_get_config_list_rejects_non_lists(config):
    with pytest.raises(ConfigError, match="is not of type list"):
        config.get_config_list("name")


def test_empty_list_yields_no_configs():
    assert Config({"items": []}).get_config_list("items") == []


###############################################################################
################################# CONVERSION ##################################
###############################################################################


def test_as_dict_returns_a_deep_copy(config):
    snapshot = config.as_dict()
    snapshot["model"]["layers"].append({"size": 99})
    snapshot["count"] = 0
    assert config.get_int("count") == 3
    assert config.as_dict()["model"]["layers"] == [{"size": 4}]


def test_constructor_unwraps_nested_config_objects():
    outer = Config({"inner": Config({"x": 1}), "many": [Config({"y": 2})]})
    assert outer.as_dict() == {"inner": {"x": 1}, "many": [{"y": 2}]}


def test_as_kwargs_wraps_dicts_and_lists_of_dicts(config):
    kwargs = config.as_kwargs()

    assert isinstance(kwargs["model"], Config)
    assert all(isinstance(item, Config) for item in kwargs["items"])
    assert kwargs["name"] == "run"
    assert kwargs["numbers"] == [1, 2, 3]
    assert kwargs["nothing"] is None


def test_as_kwargs_wraps_dictionaries_nested_inside_lists(config):
    layers = config.as_kwargs()["model"].as_kwargs()["layers"]
    assert isinstance(layers[0], Config)
    assert layers[0].get_int("size") == 4


def test_as_kwargs_does_not_alias_the_internal_state(config):
    kwargs = config.as_kwargs()
    kwargs["numbers"].append(4)
    assert config.as_dict()["numbers"] == [1, 2, 3]


###############################################################################
################################### UPDATING ##################################
###############################################################################


def test_update_with_dict_adds_missing_keys_and_merges_nested_dicts(config):
    config.update_with_dict({"extra": 1, "model": {"width": 8}})
    assert config.get_int("extra") == 1
    assert config.section("model").as_dict() == {
        "model_type": "mlp",
        "layers": [{"size": 4}],
        "dropout": 0.1,
        "width": 8,
    }


def test_update_with_dict_accepts_equal_values(config):
    config.update_with_dict({"count": 3})
    assert config.get_int("count") == 3


def test_update_with_dict_rejects_conflicting_values_and_keeps_state(config):
    before = config.as_dict()
    with pytest.raises(ConfigError, match="Conflict for key 'count'"):
        config.update_with_dict({"count": 4})
    assert config.as_dict() == before


def test_update_with_another_config(config):
    config.update(Config({"extra": "x"}))
    assert config.get_str("extra") == "x"


def test_update_does_not_alias_the_other_config():
    target, other = Config({"a": 1}), Config({"nested": {"k": 1}})
    target.update(other)
    other.update_with_dict({"nested": {"added": 2}})
    assert target.as_dict() == {"a": 1, "nested": {"k": 1}}


###############################################################################
################################### SAVING ####################################
###############################################################################


def test_save_writes_yaml_that_loads_back_equal(config, tmp_path):
    path = config.save(tmp_path / "out.yaml")
    assert path == tmp_path / "out.yaml"
    assert yaml.safe_load(path.read_text()) == config.as_dict()


def test_save_refuses_to_overwrite(config, tmp_path):
    config.save(tmp_path / "out.yaml")
    with pytest.raises(ConfigError, match="already exists"):
        config.save(tmp_path / "out.yaml")
