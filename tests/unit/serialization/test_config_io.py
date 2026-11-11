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

"""Unit tests for raw configuration I/O: load, save, copy and round-trips."""

import pytest
import yaml

from xanesnet.serialization.config import copy_raw_config, load_raw_config, merge_raw_configs, save_raw_config
from xanesnet.utils.exceptions import ConfigError

###############################################################################
################################ load_raw_config ##############################
###############################################################################


@pytest.mark.parametrize("name", ["config.yaml", "config.yml"])
def test_load_returns_the_parsed_dictionary(write_yaml, train_config, name):
    path = write_yaml(train_config, name=name)
    assert load_raw_config(path) == train_config


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_raw_config(tmp_path / "missing.yaml")


def test_load_invalid_yaml_raises(write_yaml):
    path = write_yaml("key: [unclosed\n")
    with pytest.raises(ConfigError, match="Error parsing YAML"):
        load_raw_config(path)


@pytest.mark.parametrize("text", ["- a\n- b\n", "just a string\n", ""], ids=["list", "string", "empty-file"])
def test_load_non_mapping_top_level_raises(write_yaml, text):
    path = write_yaml(text)
    with pytest.raises(ConfigError, match="top-level dictionary"):
        load_raw_config(path)


###############################################################################
################################ save_raw_config ##############################
###############################################################################


def test_save_writes_a_yaml_file_and_returns_its_path(tmp_path, train_config):
    destination = tmp_path / "saved.yaml"
    assert save_raw_config(train_config, destination) == destination
    assert yaml.safe_load(destination.read_text()) == train_config


@pytest.mark.parametrize("name", ["config.txt", "config.json", "config"])
def test_save_rejects_wrong_extension(tmp_path, name):
    with pytest.raises(ConfigError, match="extension"):
        save_raw_config({"seed": 1}, tmp_path / name)
    assert not (tmp_path / name).exists()


def test_save_rejects_missing_parent_directory(tmp_path):
    with pytest.raises(ConfigError, match="Directory for config file does not exist"):
        save_raw_config({"seed": 1}, tmp_path / "no_such_dir" / "config.yaml")


def test_save_refuses_to_overwrite_and_keeps_original(tmp_path):
    destination = save_raw_config({"seed": 1}, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="already exists"):
        save_raw_config({"seed": 2}, destination)
    assert yaml.safe_load(destination.read_text()) == {"seed": 1}


###############################################################################
################################ copy_raw_config ##############################
###############################################################################


@pytest.mark.parametrize("new_name, expected", [(None, "source.yaml"), ("renamed.yaml", "renamed.yaml")])
def test_copy_keeps_or_changes_the_filename(write_yaml, tmp_path, new_name, expected):
    source = write_yaml({"seed": 1}, name="source.yaml")
    destination_dir = tmp_path / "out"
    destination_dir.mkdir()

    result = copy_raw_config(source, destination_dir, new_name=new_name)

    assert result == destination_dir / expected
    assert load_raw_config(result) == {"seed": 1}
    assert source.is_file(), "the source file must be left in place"


def test_copy_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        copy_raw_config(tmp_path / "missing.yaml", tmp_path)


def test_copy_rejects_non_yaml_suffix(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("seed: 1\n")
    with pytest.raises(ValueError, match="suffix"):
        copy_raw_config(source, tmp_path)


###############################################################################
################################## ROUND TRIP #################################
###############################################################################


def test_save_then_load_round_trip_preserves_values_and_key_order(tmp_path):
    config = {
        "zeta": 1,
        "none": None,
        "auto": "auto",
        "flag": False,
        "small": 1e-3,
        "nested": {"values": [1, 2.5, "x", None], "empty": {}},
    }
    saved = save_raw_config(config, tmp_path / "round_trip.yaml")

    loaded = load_raw_config(saved)

    assert loaded == config
    assert list(loaded) == list(config)


###############################################################################
################################### MERGING ###################################
###############################################################################


def test_merge_unites_keys_and_merges_nested_dicts_recursively():
    base = {"seed": 1, "model": {"model_type": "mlp", "in_size": 4}}
    overlay = {"seed": 1, "model": {"model_type": "mlp", "out_size": 8}, "device": "cpu"}
    assert merge_raw_configs(base, overlay) == {
        "seed": 1,
        "model": {"model_type": "mlp", "in_size": 4, "out_size": 8},
        "device": "cpu",
    }


@pytest.mark.parametrize(
    "base, overlay, key",
    [
        ({"a": 1}, {"a": 2}, "a"),
        ({"model": {"head": {"hidden_size": 1}}}, {"model": {"head": {"hidden_size": 2}}}, "hidden_size"),
        ({"model": {"model_type": "mlp"}}, {"model": {"model_type": "MLP"}}, "model_type"),
    ],
    ids=["scalar", "deeply-nested", "case-sensitive"],
)
def test_merge_conflict_raises_and_names_the_key(base, overlay, key):
    with pytest.raises(ConfigError, match=f"Conflict for key '{key}'"):
        merge_raw_configs(base, overlay)

