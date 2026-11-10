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

"""Unit tests: invalid configs must be rejected with a ``ConfigError``"""

import pytest

from xanesnet.serialization.config import merge_raw_configs
from xanesnet.serialization.schema_validation import validate_config_schema
from xanesnet.utils.exceptions import ConfigError

###############################################################################
############################## BASELINES ACCEPTED #############################
###############################################################################


def test_minimal_configs_are_accepted(train_config, infer_user_config, signature, analyze_config):
    validate_config_schema(train_config, "train")
    validate_config_schema(merge_raw_configs(infer_user_config, signature), "infer")
    validate_config_schema(analyze_config, "analyze")


###############################################################################
################################ TRAIN REJECTIONS #############################
###############################################################################

# name -> (mutation, regex that the error message must match)
TRAIN_REJECTIONS = {
    "unknown_top_level_key": (lambda c: c.update(foo=1), r"config:.*foo.*unexpected"),
    "inferencer_section_in_train": (lambda c: c.update(inferencer={"inferencer_type": "basic"}), r"config:"),
    "seed_is_a_string": (lambda c: c.update(seed="abc"), r"config\.seed"),
    "unknown_datasource_type": (lambda c: c["datasource"].update(datasource_type="nope"), r"config\.datasource"),
    "unknown_dataset_type": (lambda c: c["dataset"].update(dataset_type="nope"), r"config\.dataset"),
    "unknown_descriptor_type": (
        lambda c: c["dataset"]["descriptors"][0].update(descriptor_type="nope"),
        r"config\.dataset",
    ),
    "unknown_model_type": (lambda c: c["model"].update(model_type="nope"), r"config\.model"),
    "model_param_wrong_type": (lambda c: c["model"].update(hidden_size="wide"), r"config\.model"),
    "model_param_out_of_range": (lambda c: c["model"].update(hidden_size=0), r"config\.model"),
    "model_unknown_key": (lambda c: c["model"].update(not_a_param=1), r"config\.model"),
    "unknown_trainer_type": (lambda c: c["trainer"].update(trainer_type="nope"), r"config\.trainer"),
    "epochs_wrong_type": (lambda c: c["trainer"].update(epochs="ten"), r"config\.trainer"),
    "epochs_zero": (lambda c: c["trainer"].update(epochs=0), r"config\.trainer"),
    "unknown_loss_type": (lambda c: c["trainer"]["loss"].append({"loss_type": "nope"}), r"config\.trainer"),
    "unknown_strategy_type": (lambda c: c["strategy"].update(strategy_type="nope"), r"config\.strategy"),
    "kfold_n_splits_too_small": (
        lambda c: c.update(strategy={"strategy_type": "kfold", "n_splits": 1}),
        r"config\.strategy",
    ),
    "unknown_encoding_type": (lambda c: c.update(encodings=[{"encoding_type": "nope"}]), r"config\.encodings"),
}


@pytest.mark.parametrize("name", TRAIN_REJECTIONS)
def test_invalid_train_config_is_rejected(name, train_config, mutate):
    mutation, expected_message = TRAIN_REJECTIONS[name]
    with pytest.raises(ConfigError, match=expected_message):
        validate_config_schema(mutate(train_config, mutation), "train")


@pytest.mark.parametrize("section", ["datasource", "dataset", "model", "trainer", "strategy"])
def test_missing_required_train_section_is_rejected(section, train_config):
    del train_config[section]
    with pytest.raises(ConfigError, match=f"'{section}' is a required property"):
        validate_config_schema(train_config, "train")


def test_error_message_names_the_mode_and_the_config_path(train_config):
    train_config["model"]["model_type"] = "nope"
    with pytest.raises(ConfigError, match=r"^Invalid train config at config\.model"):
        validate_config_schema(train_config, "train")


###############################################################################
################################ AUTO TOKEN RULES #############################
###############################################################################


@pytest.mark.parametrize("token", ["auto", "AUTO"])
def test_auto_is_accepted_case_insensitively_where_allowed(train_config, token):
    train_config["model"]["in_size"] = token
    train_config["model"]["out_size"] = token
    validate_config_schema(train_config, "train")


def test_auto_is_rejected_for_fields_that_cannot_be_resolved(train_config):
    train_config["model"]["hidden_size"] = "auto"
    with pytest.raises(ConfigError, match=r"config\.model"):
        validate_config_schema(train_config, "train")


###############################################################################
############################### INFER REJECTIONS ##############################
###############################################################################

INFER_REJECTIONS = {
    "trainer_section_in_infer": (lambda c: c.update(trainer={"trainer_type": "basic"}), r"config:"),
    "missing_inferencer": (lambda c: c.pop("inferencer"), r"'inferencer' is a required property"),
    "missing_model": (lambda c: c.pop("model"), r"'model' is a required property"),
    "unknown_inferencer_type": (lambda c: c["inferencer"].update(inferencer_type="nope"), r"config\.inferencer"),
    "unknown_model_type": (lambda c: c["model"].update(model_type="nope"), r"config\.model"),
}


@pytest.mark.parametrize("name", INFER_REJECTIONS)
def test_invalid_infer_config_is_rejected(name, infer_user_config, signature, mutate):
    mutation, expected_message = INFER_REJECTIONS[name]
    invalid = mutate(merge_raw_configs(infer_user_config, signature), mutation)
    with pytest.raises(ConfigError, match=expected_message):
        validate_config_schema(invalid, "infer")


def test_user_infer_config_without_the_checkpoint_signature_is_rejected(infer_user_config):
    """The ``infer`` mode validates the *merged* config, so the bare user YAML lacks model and encodings."""
    with pytest.raises(ConfigError, match="is a required property"):
        validate_config_schema(infer_user_config, "infer")


###############################################################################
############################### ANALYZE REJECTIONS ############################
###############################################################################

ANALYZE_REJECTIONS = {
    "unknown_top_level_key": (lambda c: c.update(foo=1), r"config:"),
    "unknown_selector_type": (lambda c: c.update(selectors=[{"selector_type": "nope"}]), r"config\.selectors"),
    "unknown_collector_type": (lambda c: c.update(collectors=[{"collector_type": "nope"}]), r"config\.collectors"),
    "unknown_plotter_type": (lambda c: c.update(plotters=[{"plotter_type": "nope"}]), r"config\.plotters"),
    "tensorboard_module_without_dirs": (
        lambda c: c.update(custom_modules=[{"custom_module_type": "tensorboard"}]),
        r"config\.custom_modules",
    ),
}


@pytest.mark.parametrize("name", ANALYZE_REJECTIONS)
def test_invalid_analyze_config_is_rejected(name, analyze_config, mutate):
    mutation, expected_message = ANALYZE_REJECTIONS[name]
    with pytest.raises(ConfigError, match=expected_message):
        validate_config_schema(mutate(analyze_config, mutation), "analyze")


def test_custom_module_names_are_not_restricted_by_the_schema(analyze_config):
    """Custom modules are open-ended: unknown names pass the schema and are resolved by the registry at runtime."""
    analyze_config["custom_modules"] = [{"custom_module_type": "my_plugin", "anything": 1}]
    validated = validate_config_schema(analyze_config, "analyze")
    assert validated["custom_modules"] == [{"custom_module_type": "my_plugin", "anything": 1}]
