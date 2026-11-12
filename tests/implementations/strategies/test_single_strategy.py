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

"""Unit tests for the ``single`` strategy (one model, one trainer)."""

from pathlib import Path

import pytest

from . import fakes


def test_the_model_is_created_from_the_model_config(build, model_config):
    strategy = build("single")

    strategy.setup_models(model_config)

    assert len(strategy.models) == 1
    assert isinstance(strategy.models[0], fakes.FakeModel)
    assert strategy.models[0].kwargs == {"model_type": fakes.MODEL_TYPE, "width": 3}


def test_the_trainer_receives_its_collaborators_and_the_trainer_config(prepared):
    strategy = prepared("single")

    strategy.setup_trainers("cpu")

    (trainer,) = strategy.trainers
    assert trainer.kwargs["trainer_type"] == fakes.TRAINER_TYPE
    assert trainer.kwargs["epochs"] == 2
    assert trainer.kwargs["model"] is strategy.models[0]
    assert trainer.kwargs["device"] == "cpu"
    assert trainer.kwargs["checkpointer"] is strategy.checkpointer
    assert trainer.kwargs["encoding"] is strategy.encoding
    assert trainer.dataset.get_subset_indices(0) == strategy.dataset.get_subset_indices(0)


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("models", "model is not initialized"),
        ("trainer_config", "no trainer config"),
        ("checkpointer", "checkpointer is not instantiated"),
        ("encoding", "encoding is not initialized"),
    ],
)
def test_trainers_need_every_prerequisite(prepared, missing, message):
    strategy = prepared("single")
    setattr(strategy, missing, [] if missing == "models" else None)

    with pytest.raises(ValueError, match=message):
        strategy.setup_trainers("cpu")


def test_training_runs_the_trainer_then_records_the_finished_member(prepared, events):
    strategy = prepared("single")
    strategy.setup_trainers("cpu")

    models = strategy.run_training()

    assert models is strategy.models
    assert [name for name, _ in events if name != "trainer_created"] == ["train", "finish_member"]


def test_tensorboard_is_only_used_when_a_directory_is_configured(prepared, tensorboard_calls, tmp_path):
    silent = prepared("single")
    silent.setup_trainers("cpu")
    silent.run_training()
    assert [name for name, _ in tensorboard_calls] == ["close"]

    tensorboard_calls.clear()
    logged = prepared("single", tensorboard_dir=tmp_path / "tb")
    logged.setup_trainers("cpu")
    logged.run_training()
    assert tensorboard_calls == [("new_run", tmp_path / "tb"), ("close", None)]


def test_tensorboard_is_closed_when_training_fails(prepared, tensorboard_calls, events):
    strategy = prepared("single")
    strategy.models[0].fail_on_train = True
    strategy.setup_trainers("cpu")

    with pytest.raises(RuntimeError, match="training failed"):
        strategy.run_training()

    assert tensorboard_calls[-1] == ("close", None)
    assert "finish_member" not in [name for name, _ in events]


@pytest.mark.parametrize("missing", ["trainers", "models", "checkpointer"])
def test_training_needs_trainers_models_and_a_checkpointer(prepared, missing):
    strategy = prepared("single")
    strategy.setup_trainers("cpu")
    setattr(strategy, missing, None if missing == "checkpointer" else [])

    with pytest.raises(ValueError, match="Cannot run training"):
        strategy.run_training()


###############################################################################
################################### INFERENCE #################################
###############################################################################


def test_the_inferencer_is_chosen_by_the_config_and_receives_the_model(build, model_config, inferencer_config):
    strategy = build("single", trainer_config=None)
    strategy.setup_encoding(fakes.ENCODING_CONFIG)
    strategy.setup_models(model_config)

    strategy.setup_inferencers("cpu")

    kwargs = strategy.inferencer.kwargs
    assert kwargs["batch_size"] == 4
    assert kwargs["model"] is strategy.models[0]
    assert kwargs["dataset"] is strategy.dataset
    assert kwargs["device"] == "cpu"
    assert kwargs["encoding"] is strategy.encoding


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("models", "model is not initialized"),
        ("inferencer_config", "no inferencer config"),
        ("encoding", "encoding is not initialized"),
    ],
)
def test_inferencers_need_every_prerequisite(prepared, missing, message):
    strategy = prepared("single")
    setattr(strategy, missing, [] if missing == "models" else None)

    with pytest.raises(ValueError, match=message):
        strategy.setup_inferencers("cpu")


def test_inference_delegates_to_the_inferencer(build, model_config, events):
    strategy = build("single", trainer_config=None)
    strategy.setup_encoding(fakes.ENCODING_CONFIG)
    strategy.setup_models(model_config)
    strategy.setup_inferencers("cpu")

    strategy.run_inference(Path("out"))

    assert ("infer", Path("out")) in events


def test_inference_needs_an_inferencer(build):
    with pytest.raises(ValueError, match="Inferencer is not initialized"):
        build("single").run_inference(None)
