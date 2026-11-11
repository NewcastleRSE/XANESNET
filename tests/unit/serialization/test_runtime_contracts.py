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

"""Unit tests for registry-backed runtime contracts that JSON Schema cannot express."""

import pytest

from xanesnet.serialization.runtime_contracts import validate_runtime_contracts
from xanesnet.utils.exceptions import ConfigError

###############################################################################
############################ DATASET / MODEL PAIRING ##########################
###############################################################################


@pytest.mark.parametrize("mode", ["train", "infer"])
def test_registered_dataset_model_pair_is_accepted(mode):
    config = {"dataset": {"dataset_type": "descriptor"}, "model": {"model_type": "mlp"}}
    validate_runtime_contracts(config, mode)


@pytest.mark.parametrize("mode", ["train", "infer"])
def test_pairing_is_case_insensitive(mode):
    config = {"dataset": {"dataset_type": "Descriptor"}, "model": {"model_type": "MLP"}}
    validate_runtime_contracts(config, mode)


@pytest.mark.parametrize("mode", ["train", "infer"])
def test_unregistered_dataset_model_pair_is_rejected(mode):
    config = {"dataset": {"dataset_type": "descriptor"}, "model": {"model_type": "schnet"}}
    with pytest.raises(ConfigError, match="No batch processor registered for dataset 'descriptor' and model 'schnet'"):
        validate_runtime_contracts(config, mode)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"dataset": {"dataset_type": "descriptor"}},
        {"model": {"model_type": "mlp"}},
        {"dataset": "not-a-section", "model": {"model_type": "mlp"}},
    ],
    ids=["empty", "no-model", "no-dataset", "non-dict-section"],
)
def test_incomplete_configs_are_left_to_the_schema(config):
    """Missing sections are reported by schema validation, not by the runtime contract."""
    validate_runtime_contracts(config, "train")


###############################################################################
############################ INFERENCE-ONLY CONTRACTS #########################
###############################################################################


def _infer_config(**overrides):
    config = {
        "dataset": {"dataset_type": "descriptor"},
        "model": {"model_type": "mlp", "in_size": 6, "out_size": 10},
        "encodings": [{"encoding_type": "identity"}],
        "strategy": {"strategy_type": "single"},
        "inferencer": {"inferencer_type": "basic"},
    }
    config.update(overrides)
    return config


def test_concrete_inference_config_is_accepted():
    validate_runtime_contracts(_infer_config(), "infer")


@pytest.mark.parametrize("token", ["auto", "AUTO", "Auto"])
def test_unresolved_model_dimension_is_rejected_for_inference(token):
    config = _infer_config(model={"model_type": "mlp", "in_size": token, "out_size": 10})
    with pytest.raises(ConfigError, match=r"unresolved automatic model value at model\.in_size"):
        validate_runtime_contracts(config, "infer")


def test_unresolved_nested_model_value_reports_its_full_path():
    config = _infer_config(model={"model_type": "mlp", "heads": [{"size": 3}, {"size": "auto"}]})
    with pytest.raises(ConfigError, match=r"model\.heads\.1\.size"):
        validate_runtime_contracts(config, "infer")


def test_unresolved_encoding_value_is_rejected_for_inference():
    config = _infer_config(encodings=[{"encoding_type": "identity"}, {"encoding_type": "gaussian", "num_points": "auto"}])
    with pytest.raises(ConfigError, match=r"unresolved automatic encoding value at encodings\.1\.num_points"):
        validate_runtime_contracts(config, "infer")


def test_auto_is_still_allowed_when_training():
    config = {"dataset": {"dataset_type": "descriptor"}, "model": {"model_type": "mlp", "in_size": "auto"}}
    validate_runtime_contracts(config, "train")


@pytest.mark.parametrize("strategy", ["deep_ensemble", "bootstrap", "kfold"])
def test_multi_model_strategies_require_the_ensemble_inferencer(strategy):
    config = _infer_config(strategy={"strategy_type": strategy}, inferencer={"inferencer_type": "basic"})
    with pytest.raises(ConfigError, match=f"'{strategy}' requires inferencer 'ensemble'"):
        validate_runtime_contracts(config, "infer")


@pytest.mark.parametrize("strategy", ["deep_ensemble", "bootstrap", "kfold"])
def test_multi_model_strategies_accept_the_ensemble_inferencer(strategy):
    validate_runtime_contracts(
        _infer_config(strategy={"strategy_type": strategy}, inferencer={"inferencer_type": "ensemble"}), "infer"
    )


def test_single_model_strategy_accepts_the_basic_inferencer():
    validate_runtime_contracts(_infer_config(strategy={"strategy_type": "single"}), "infer")


###############################################################################
################################ OTHER MODES ##################################
###############################################################################


def test_analyze_mode_has_no_runtime_contracts():
    validate_runtime_contracts({"anything": "goes"}, "analyze")


def test_unknown_mode_is_rejected():
    with pytest.raises(ConfigError, match="Unknown config validation mode"):
        validate_runtime_contracts({}, "nope")  # type: ignore[arg-type]
