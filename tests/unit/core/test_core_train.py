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

"""Unit tests for the ``core_train.train`` pipeline, run against registered fakes."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from xanesnet import core_train
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config, load_raw_config

from . import fakes


@pytest.fixture(autouse=True)
def _stub_collaborators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace collaborators that have their own unit tests or need real models."""
    monkeypatch.setattr(core_train, "resolve_auto_model_config", lambda config, dataset, encoding: config)
    monkeypatch.setattr(core_train, "_summary_models", lambda *args: None)


@pytest.fixture(autouse=True)
def _run_directory(tmp_path: Path) -> None:
    """``train`` expects the CLI to have created the run sub-folders already."""
    (tmp_path / "models").mkdir()
    (tmp_path / "checkpoints").mkdir()


@pytest.fixture
def config() -> Config:
    return Config(
        {
            "seed": 1,
            "device": "cpu",
            "datasource": {"datasource_type": fakes.DATASOURCE_TYPE, "path": "somewhere"},
            "dataset": {"dataset_type": fakes.DATASET_TYPE, "root": "dataset_root"},
            "strategy": {"strategy_type": fakes.STRATEGY_TYPE, "option": 3},
            "trainer": {"trainer_type": "basic", "epochs": 2},
            "model": {"model_type": "fake", "hidden_size": 8},
        }
    )


@pytest.fixture
def args() -> Namespace:
    return Namespace(tensorboard=False, dry_run=False, restart_training_state=False)


def training_state() -> dict:
    """Build a minimal checkpoint training-state payload.

    Returns:
        Optimizer and scheduler state mappings for one trainer.
    """
    return {
        "optimizer_states": [{"lr": 0.1}],
        "epoch_scheduler_states": [{"epoch": 4}],
        "warmup_scheduler_states": [{"warmup": 1}],
    }


def make_checkpoint(with_training_state: bool = True) -> Checkpoint:
    """Build a checkpoint for the fake training pipeline.

    Args:
        with_training_state: Whether to include optimizer and scheduler state.

    Returns:
        Checkpoint containing the fake signature, model state, and optional
        training state.
    """
    return Checkpoint(
        signature=Config(fakes.SIGNATURE),
        model_states=[{"weight": 1}],
        training_state=training_state() if with_training_state else None,
    )


###############################################################################
################################# ORCHESTRATION ###############################
###############################################################################


def test_pipeline_stages_run_in_order(config, args, tmp_path):
    core_train.train(config, args, tmp_path, None)

    stages = [
        name
        for name in fakes.event_names()
        if name
        in {
            "datasource",
            "dataset",
            "strategy",
            "setup_model_splits",
            "setup_encoding",
            "setup_models",
            "setup_checkpointer",
            "init_model_weights",
            "setup_trainers",
            "run_training",
            "checkpointer_save",
        }
    ]
    assert stages == [
        "datasource",
        "dataset",
        "strategy",
        "setup_model_splits",
        "setup_encoding",
        "setup_models",
        "setup_checkpointer",
        "init_model_weights",
        "setup_trainers",
        "run_training",
        "checkpointer_save",
    ]


def test_dataset_is_prepared_and_preloaded_before_the_strategy_sees_it(config, args, tmp_path):
    core_train.train(config, args, tmp_path, None)

    names = fakes.event_names()
    assert names.index("prepare") < names.index("check_preload") < names.index("strategy")


def test_components_receive_their_configuration(config, args, tmp_path):
    core_train.train(config, args, tmp_path, None)

    assert fakes.payload("datasource")["path"] == "somewhere"
    assert fakes.payload("dataset")["root"] == "dataset_root"

    strategy_kwargs = fakes.payload("strategy")
    assert strategy_kwargs["option"] == 3
    assert strategy_kwargs["checkpoint_dir"] == tmp_path / "checkpoints"
    assert strategy_kwargs["trainer_config"].get_int("epochs") == 2
    assert strategy_kwargs["inferencer_config"] is None

    assert fakes.payload("setup_models").get_int("hidden_size") == 8
    assert fakes.payload("setup_trainers") == "cpu"


@pytest.mark.parametrize("tensorboard", [True, False])
def test_tensorboard_directory_follows_the_cli_flag(config, args, tmp_path, tensorboard):
    args.tensorboard = tensorboard

    core_train.train(config, args, tmp_path, None)

    expected = tmp_path / "tensorboard" if tensorboard else None
    assert fakes.payload("strategy")["tensorboard_dir"] == expected


###############################################################################
################################### ARTIFACTS #################################
###############################################################################


def test_resolved_config_includes_the_encoding_signature(config, args, tmp_path):
    core_train.train(config, args, tmp_path, None)

    resolved = load_raw_config(tmp_path / "resolved_train_config.yaml")
    assert resolved["encodings"] == [{"encoding_type": "fake_encoding", "num_points": 3}]
    assert resolved["model"]["hidden_size"] == 8


def test_signature_splits_and_final_checkpoint_locations(config, args, tmp_path):
    core_train.train(config, args, tmp_path, None)

    assert load_raw_config(tmp_path / "models" / "signature.yaml")["model"] == {"model_type": "fake"}
    splits = json.loads((tmp_path / "split_indices.json").read_text())
    assert splits == {"models": [{"train": [0, 1, 2], "valid": [3]}]}
    assert fakes.payload("checkpointer_save") == tmp_path / "models" / "final.pth"


def test_trained_models_are_moved_to_cpu(config, args, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(core_train, "_save_results", lambda model_list, *rest: seen.extend(model_list))

    core_train.train(config, args, tmp_path, None)

    assert [model.device.type for model in seen] == ["cpu"]


###############################################################################
############################### CHECKPOINT RESTORE ############################
###############################################################################


def test_no_checkpoint_restores_nothing(config, args, tmp_path):
    core_train.train(config, args, tmp_path, None)

    assert not any(
        name in fakes.event_names() for name in ("set_state_dicts", "load_optimizer", "load_epoch_scheduler")
    )


def test_checkpoint_restores_models_optimizer_and_schedulers(config, args, tmp_path):
    core_train.train(config, args, tmp_path, make_checkpoint())

    assert fakes.payload("set_state_dicts") == [{"weight": 1}]
    assert fakes.payload("load_optimizer") == {"lr": 0.1}
    assert fakes.payload("load_epoch_scheduler") == {"epoch": 4}
    assert fakes.payload("load_warmup_scheduler") == {"warmup": 1}
    names = fakes.event_names()
    assert names.index("setup_trainers") < names.index("set_state_dicts") < names.index("run_training")


def test_restart_training_state_restores_only_the_model_weights(config, args, tmp_path):
    args.restart_training_state = True

    core_train.train(config, args, tmp_path, make_checkpoint())

    assert "set_state_dicts" in fakes.event_names()
    assert not any(name.startswith("load_") for name in fakes.event_names())


def test_checkpoint_without_training_state_restores_only_the_model_weights(config, args, tmp_path):
    core_train.train(config, args, tmp_path, make_checkpoint(with_training_state=False))

    assert "set_state_dicts" in fakes.event_names()
    assert not any(name.startswith("load_") for name in fakes.event_names())


def test_missing_warmup_scheduler_state_is_skipped(config, args, tmp_path):
    checkpoint = make_checkpoint()
    assert checkpoint.training_state is not None
    checkpoint.training_state["warmup_scheduler_states"] = [None]

    core_train.train(config, args, tmp_path, checkpoint)

    assert "load_epoch_scheduler" in fakes.event_names()
    assert "load_warmup_scheduler" not in fakes.event_names()


###############################################################################
################################### DRY RUN ###################################
###############################################################################


@pytest.mark.parametrize("dry_run", [True, False])
def test_dry_run_flag_is_forwarded_to_result_saving(config, args, tmp_path, monkeypatch, dry_run):
    args.dry_run = dry_run
    forwarded = []
    monkeypatch.setattr(core_train, "_save_results", lambda *params: forwarded.append(params[-1]))

    core_train.train(config, args, tmp_path, None)

    assert forwarded == [dry_run]


def test_dry_run_resets_peak_memory_statistics_before_training(config, args, tmp_path, monkeypatch):
    args.dry_run = True
    monkeypatch.setattr(
        core_train,
        "reset_peak_memory_stats",
        lambda device: fakes.EVENTS.append(("reset_peak", device)),
    )
    monkeypatch.setattr(core_train, "_save_results", lambda *params: None)

    core_train.train(config, args, tmp_path, None)

    names = fakes.event_names()
    assert names.index("reset_peak") < names.index("run_training")
    assert fakes.payload("reset_peak") == "cpu"


def test_peak_memory_is_untouched_outside_dry_runs(config, args, tmp_path, monkeypatch):
    monkeypatch.setattr(
        core_train,
        "reset_peak_memory_stats",
        lambda device: fakes.EVENTS.append(("reset_peak", device)),
    )

    core_train.train(config, args, tmp_path, None)

    assert "reset_peak" not in fakes.event_names()


###############################################################################
################################### FAILURES ##################################
###############################################################################


@pytest.mark.parametrize("section", ["datasource", "dataset", "strategy"])
def test_unregistered_component_types_fail_before_any_output_is_written(config, args, tmp_path, section):
    raw = config.as_dict()
    raw[section][f"{section}_type"] = "does_not_exist"

    with pytest.raises(KeyError, match="does_not_exist"):
        core_train.train(Config(raw), args, tmp_path, None)

    assert not [path for path in tmp_path.rglob("*") if path.is_file()]
