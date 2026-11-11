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

from xanesnet.serialization.config import merge_raw_configs
from xanesnet.serialization.schema_validation import validate_config_schema

###############################################################################
######################## Test DEFAULTS OF THE SCHEMAS #########################
###############################################################################


def test_train_defaults_are_applied(train_config):
    validated = validate_config_schema(train_config, "train")

    assert validated["seed"] is None
    assert validated["device"] == "cpu"
    assert validated["dataset"]["preload"] is True
    assert validated["dataset"]["split_ratios"] == [1.0]
    assert validated["model"]["dropout"] == 0.1
    assert validated["trainer"]["shuffle"] is True
    assert validated["trainer"]["warmup_steps"] == 500
    assert validated["trainer"]["lr_scheduler"] == {"lr_scheduler_type": "none", "last_epoch": -1}
    assert validated["trainer"]["early_stopper"] == {"early_stopper_type": "none", "restore_best": True}
    assert validated["strategy"]["weight_init"] == "default"
    assert validated["strategy"]["bias_init"] == "zeros"


def test_omitted_optional_sections_get_their_defaults(train_config):
    del train_config["encodings"]
    del train_config["trainer"]["loss"]
    train_config["strategy"] = {"strategy_type": "kfold"}

    validated = validate_config_schema(train_config, "train")

    assert validated["encodings"] == [{"encoding_type": "identity"}]
    assert validated["trainer"]["loss"] == [{"loss_type": "mse"}]
    assert validated["strategy"]["n_splits"] == 3
    assert validated["strategy"]["n_repeats"] == 1


def test_infer_defaults_are_applied(infer_user_config, signature):
    validated = validate_config_schema(merge_raw_configs(infer_user_config, signature), "infer")

    assert validated["device"] == "cpu"
    assert validated["inferencer"]["shuffle"] is False
    assert validated["inferencer"]["num_workers"] == 0
    assert validated["inferencer"]["buffer_size"] == 100000


def test_analyze_defaults_are_applied(analyze_config):
    validated = validate_config_schema(analyze_config, "analyze")

    assert validated["preload"] is False
    for section in ("aggregators", "reporters", "plotters", "custom_modules"):
        assert validated[section] == []
    assert validated["collectors"][0]["energy_resolved"] is False


###############################################################################
############################### PRECEDENCE RULES ##############################
###############################################################################


def test_explicit_values_override_defaults(train_config):
    train_config["seed"] = 7
    train_config["trainer"]["shuffle"] = False
    train_config["model"]["dropout"] = 0.0

    validated = validate_config_schema(train_config, "train")

    assert validated["seed"] == 7
    assert validated["trainer"]["shuffle"] is False
    assert validated["model"]["dropout"] == 0.0


def test_none_is_not_replaced_by_the_default(train_config):
    train_config["device"] = None
    assert validate_config_schema(train_config, "train")["device"] is None
