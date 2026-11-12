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

"""Unit tests for the behaviour every strategy inherits from ``Strategy`` (exercised through ``single``)."""

import logging
from types import SimpleNamespace

import pytest
import torch

from xanesnet.utils.exceptions import ConfigError

from . import fakes


def test_a_strategy_needs_a_trainer_or_an_inferencer_config(build):
    with pytest.raises(ValueError, match="Either trainer_config or inferencer_config"):
        build("single", trainer_config=None, inferencer_config=None)

    assert build("single", trainer_config=None).trainer_config is None
    assert build("single", inferencer_config=None).inferencer_config is None


###############################################################################
################################## MODEL STATE ################################
###############################################################################


def test_weights_are_initialised_with_the_configured_schemes_on_every_model(build, model_config):
    strategy = build("deep_ensemble", n_models=2)
    strategy.setup_models(model_config)

    strategy.init_model_weights()

    assert [model.init_calls for model in strategy.models] == [[("xavier", "zeros", {"gain": 2.0})]] * 2


def test_weights_cannot_be_initialised_before_the_models_exist(build):
    with pytest.raises(ValueError, match="models are not initialized"):
        build("single").init_model_weights()


def test_state_dicts_are_loaded_into_the_models_in_order(build, model_config):
    strategy = build("deep_ensemble", n_models=2)
    strategy.setup_models(model_config)
    states = [{"linear.weight": torch.full((1, 1), 1.0), "linear.bias": torch.zeros(1)},
              {"linear.weight": torch.full((1, 1), 2.0), "linear.bias": torch.zeros(1)}]

    strategy.set_state_dicts(states)

    assert [model.linear.weight.item() for model in strategy.models] == [1.0, 2.0]


def test_too_few_checkpoint_states_are_an_error(build, model_config):
    strategy = build("deep_ensemble", n_models=2)
    strategy.setup_models(model_config)

    with pytest.raises(ConfigError, match="1 model states; strategy requires 2"):
        strategy.set_state_dicts([strategy.models[0].state_dict()])


def test_extra_checkpoint_states_are_ignored_with_a_warning(build, model_config, caplog):
    strategy = build("single")
    strategy.setup_models(model_config)
    first = {"linear.weight": torch.full((1, 1), 5.0), "linear.bias": torch.zeros(1)}
    extra = {"linear.weight": torch.full((1, 1), 9.0), "linear.bias": torch.zeros(1)}

    with caplog.at_level(logging.WARNING):
        strategy.set_state_dicts([first, extra])

    assert strategy.models[0].linear.weight.item() == 5.0
    assert "1 extra model states" in caplog.text


def test_state_dicts_cannot_be_loaded_before_the_models_exist(build):
    with pytest.raises(ValueError, match="models are not initialized"):
        build("single").set_state_dicts([{}])


###############################################################################
################################### SPLITS ####################################
###############################################################################


def test_splits_come_from_the_dataset(build):
    dataset = fakes.FakeDataset(size=6, plans=[[[0, 1, 2, 3], [4, 5]]])

    assert build("single", dataset).setup_model_splits() == [[[0, 1, 2, 3], [4, 5]]]
    assert dataset.get_subset_indices(1) == [4, 5]


###############################################################################
################################### ENCODING ##################################
###############################################################################


def test_the_encoding_is_built_from_the_config_and_prepared_for_the_energy_grid(build):
    strategy = build("single")
    strategy.setup_model_splits()

    encoding = strategy.setup_encoding(fakes.ENCODING_CONFIG)

    assert strategy.encoding is encoding
    assert encoding.output_size(fakes.N_ENERGIES) == fakes.N_ENERGIES
    assert [config.get_str("encoding_type") for config in encoding.signature] == ["none"]


def test_training_requires_the_splits_before_the_encoding(build):
    with pytest.raises(ValueError, match="before model splits are initialized"):
        build("single", fakes.FakeDataset(plans=[])).setup_encoding(fakes.ENCODING_CONFIG)


def test_inference_can_build_the_encoding_without_splits(build):
    strategy = build("single", fakes.FakeDataset(plans=[]), trainer_config=None)
    assert strategy.setup_encoding(fakes.ENCODING_CONFIG) is strategy.encoding


class DatasetWithoutGrid(fakes.FakeDataset):
    def __init__(self, energies) -> None:
        super().__init__()
        self.energies = energies

    def __getitem__(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(energies=self.energies)


@pytest.mark.parametrize("energies", [None, torch.tensor(1.0)])
def test_samples_without_an_energy_grid_are_rejected(build, energies):
    strategy = build("single", DatasetWithoutGrid(energies))
    strategy.setup_model_splits()

    with pytest.raises(ValueError, match="spectral energy grid"):
        strategy.setup_encoding(fakes.ENCODING_CONFIG)


###############################################################################
################################## CHECKPOINTER ###############################
###############################################################################


def test_checkpointer_signature_combines_dataset_model_and_encoding(build, model_config, tmp_path):
    strategy = build("single", checkpoint_dir=tmp_path / "ckpt", checkpoint_interval=5)
    strategy.setup_model_splits()
    strategy.setup_encoding(fakes.ENCODING_CONFIG)
    strategy.setup_models(model_config)

    checkpointer = strategy.setup_checkpointer()

    assert strategy.checkpointer is checkpointer
    assert checkpointer.save_dir == tmp_path / "ckpt"
    assert checkpointer.save_interval == 5
    signature = checkpointer.signature.as_dict()
    assert signature["dataset"] == {"dataset_type": "fake"}
    assert signature["model"] == {"model_type": fakes.MODEL_TYPE, "width": 3}
    assert signature["encodings"] == [{"encoding_type": "none"}]


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ((), "encoding is not initialized"),
        (("encoding",), "Models are not initialized"),
    ],
)
def test_checkpointer_needs_encoding_and_models(build, steps, message):
    strategy = build("single")
    strategy.setup_model_splits()
    if "encoding" in steps:
        strategy.setup_encoding(fakes.ENCODING_CONFIG)

    with pytest.raises(ValueError, match=message):
        strategy.setup_checkpointer()


def test_checkpointer_is_only_for_training(build, model_config):
    strategy = build("single", trainer_config=None)
    strategy.setup_encoding(fakes.ENCODING_CONFIG)
    strategy.setup_models(model_config)

    with pytest.raises(ValueError, match="no trainer config"):
        strategy.setup_checkpointer()


def test_the_checkpointer_sees_the_current_trainers(prepared):
    strategy = prepared("single")
    strategy.setup_checkpointer()
    strategy.setup_trainers("cpu")

    assert strategy.checkpointer.trainer_provider() is strategy.trainers


def test_model_signature_requires_models(build):
    with pytest.raises(ValueError, match="Models are not initialized"):
        _ = build("single").model_signature


def test_model_signature_is_that_of_the_first_model(build, model_config):
    strategy = build("deep_ensemble", n_models=2)
    strategy.setup_models(model_config)

    assert strategy.model_signature.as_dict() == {"model_type": fakes.MODEL_TYPE, "width": 3}
