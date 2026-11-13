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

"""Unit tests for the lifecycle shared by the multi-model strategies (``deep_ensemble``, ``bootstrap``, ``kfold``)."""

from pathlib import Path

import pytest

from . import fakes

N_MODELS = 3

# strategy key -> (constructor arguments, tensorboard sub-directory prefix)
STRATEGIES = {
    "deep_ensemble": ({"n_models": N_MODELS}, "model_"),
    "bootstrap": ({"n_models": N_MODELS, "sample_fraction": 1.0}, "model_"),
    "kfold": ({"n_splits": N_MODELS, "n_repeats": 1}, "fold_"),
}


@pytest.fixture(params=list(STRATEGIES))
def key(request) -> str:
    return request.param


@pytest.fixture
def extra(key):
    return STRATEGIES[key][0]


@pytest.fixture
def tb_prefix(key):
    return STRATEGIES[key][1]


def ready(prepared, key: str, extra: dict, **kwargs):
    return prepared(key, dataset=fakes.FakeDataset(size=9), **extra, **kwargs)


def test_one_independent_model_is_created_per_member(build, key, extra, model_config):
    strategy = build(key, **extra)

    strategy.setup_models(model_config)

    assert strategy.n_models == N_MODELS
    assert len(strategy.models) == N_MODELS
    assert len({id(model) for model in strategy.models}) == N_MODELS
    assert all(model.kwargs == {"model_type": fakes.MODEL_TYPE, "width": 3} for model in strategy.models)


def test_each_trainer_gets_its_own_model_and_its_own_split(prepared, key, extra):
    strategy = ready(prepared, key, extra)

    strategy.setup_trainers("cpu")

    assert len(strategy.trainers) == N_MODELS
    for index, trainer in enumerate(strategy.trainers):
        assert trainer.model is strategy.models[index]
        assert trainer.kwargs["device"] == "cpu"
        assert trainer.kwargs["epochs"] == 2
        assert trainer.dataset.get_subset_indices(0) == strategy.dataset.plans[index][0]


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("models", "models are not initialized"),
        ("trainer_config", "no trainer config"),
        ("checkpointer", "checkpointer is not instantiated"),
        ("encoding", "encoding is not initialized"),
    ],
)
def test_trainers_need_every_prerequisite(prepared, key, extra, missing, message):
    strategy = ready(prepared, key, extra)
    setattr(strategy, missing, [] if missing == "models" else None)

    with pytest.raises(ValueError, match=message):
        strategy.setup_trainers("cpu")


def test_members_are_trained_in_order_and_each_is_recorded_as_finished(prepared, key, extra, events):
    strategy = ready(prepared, key, extra)
    strategy.setup_trainers("cpu")
    trainers = list(strategy.trainers)

    models = strategy.run_training()

    assert models is strategy.models
    sequence = [(name, trainer) for name, trainer in events if name in {"train", "finish_member"}]
    assert sequence == [(name, trainer) for trainer in trainers for name in ("train", "finish_member")]


def test_finished_trainers_are_released(prepared, key, extra):
    strategy = ready(prepared, key, extra)
    strategy.setup_trainers("cpu")

    strategy.run_training()

    assert strategy.trainers == [None] * N_MODELS


def test_every_member_logs_to_its_own_tensorboard_directory(
    prepared, key, extra, tb_prefix, tensorboard_calls, tmp_path
):
    strategy = ready(prepared, key, extra, tensorboard_dir=tmp_path / "tb")
    strategy.setup_trainers("cpu")

    strategy.run_training()

    expected = []
    for index in range(N_MODELS):
        expected += [("new_run", tmp_path / "tb" / f"{tb_prefix}{index}"), ("close", None)]
    assert tensorboard_calls == expected


def test_a_failing_member_stops_training_but_is_still_cleaned_up(prepared, key, extra, events, tensorboard_calls):
    strategy = ready(prepared, key, extra)
    strategy.models[0].fail_on_train = True
    strategy.setup_trainers("cpu")
    second_trainer = strategy.trainers[1]

    with pytest.raises(RuntimeError, match="training failed"):
        strategy.run_training()

    assert strategy.trainers[0] is None
    assert strategy.trainers[1] is second_trainer
    assert [name for name, _ in events if name == "train"] == ["train"]
    assert "finish_member" not in [name for name, _ in events]
    assert tensorboard_calls[-1] == ("close", None)


@pytest.mark.parametrize("problem", ["no_models", "missing_trainer", "no_checkpointer"])
def test_training_needs_a_trainer_for_every_model(prepared, key, extra, problem):
    strategy = ready(prepared, key, extra)
    strategy.setup_trainers("cpu")
    if problem == "no_models":
        strategy.models = []
    elif problem == "missing_trainer":
        strategy.trainers[1] = None
    else:
        strategy.checkpointer = None

    with pytest.raises(ValueError, match="Cannot run training"):
        strategy.run_training()


###############################################################################
################################### INFERENCE #################################
###############################################################################


def test_inference_uses_an_ensemble_inferencer_over_all_models(prepared, key, extra, ensemble_inferencers, events):
    strategy = ready(prepared, key, extra)

    strategy.setup_inferencers("cpu")
    strategy.run_inference(Path("out"))

    ((inferencer_key, kwargs),) = ensemble_inferencers
    assert inferencer_key == "ensemble"
    assert kwargs["models"] is strategy.models
    assert kwargs["dataset"] is strategy.dataset
    assert kwargs["encoding"] is strategy.encoding
    assert kwargs["batch_size"] == 4
    assert ("infer", Path("out")) in events


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("models", "models are not initialized"),
        ("inferencer_config", "no inferencer config"),
        ("encoding", "encoding is not initialized"),
    ],
)
def test_inferencers_need_every_prerequisite(prepared, key, extra, ensemble_inferencers, missing, message):
    strategy = ready(prepared, key, extra)
    setattr(strategy, missing, [] if missing == "models" else None)

    with pytest.raises(ValueError, match=message):
        strategy.setup_inferencers("cpu")


def test_inference_needs_an_inferencer(build, key, extra):
    with pytest.raises(ValueError, match="Inferencer is not initialized"):
        build(key, **extra).run_inference(None)
