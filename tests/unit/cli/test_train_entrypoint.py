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

"""Unit tests for the ``xanesnet train`` entry point (argument parsing and run orchestration).

The heavy ``train`` pipeline is replaced by a recorder, so these tests cover everything the entry point itself is
responsible for: loading and validating configs, merging checkpoint signatures, creating the run directory and
dispatching to the right backend.
"""

import copy
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from xanesnet import train as train_cli
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import tb_logger
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.lightning import LIGHTNING_CONTEXT_ENV
from xanesnet.utils.prompts import is_auto_yes_enabled


class Recorder:
    """Captures what the entry point passes to the core ``train`` function."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.auto_yes_during_call: list[bool] = []

    def __call__(self, config: Config, args: Namespace, save_dir: Path, checkpoint: Checkpoint | None) -> None:
        self.calls.append({"config": config, "args": args, "save_dir": save_dir, "checkpoint": checkpoint})
        self.auto_yes_during_call.append(is_auto_yes_enabled())

    @property
    def call(self) -> dict:
        assert len(self.calls) == 1, f"expected exactly one core call, got {len(self.calls)}"
        return self.calls[0]


@pytest.fixture(autouse=True)
def _reset_tensorboard():
    yield
    tb_logger.close()
    tb_logger.set_log_weights(False)


@pytest.fixture
def core(monkeypatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(train_cli, "train", recorder)
    return recorder


@pytest.fixture
def config_file(write_yaml, train_config) -> Path:
    return write_yaml(train_config, name="train.yaml")


@pytest.fixture
def out_dir(tmp_path) -> Path:
    return tmp_path / "runs"


def run(config_file: Path, out_dir: Path, *extra: str) -> None:
    train_cli.main(["-i", str(config_file), "-o", str(out_dir), "-n", "unit", *extra])


###############################################################################
############################### ARGUMENT PARSING ##############################
###############################################################################


def test_defaults_for_optional_arguments():
    args = train_cli.parse_args(["-i", "config.yaml"])

    assert args.in_file == "config.yaml"
    assert args.model_path is None
    assert args.out_dir is None
    assert args.name is None
    for flag in ("restart_training_state", "tensorboard", "log_weights", "dry_run", "yes"):
        assert getattr(args, flag) is False


def test_all_arguments_are_parsed_in_their_short_and_long_forms():
    short = train_cli.parse_args(["-i", "c.yaml", "-m", "m.pth", "-o", "out", "-n", "x", "-t", "-y"])
    long = train_cli.parse_args(
        ["--in_file", "c.yaml", "--model", "m.pth", "--out_dir", "out", "--name", "x", "--tensorboard", "--yes"]
    )

    assert vars(short) == vars(long)
    assert (short.model_path, short.out_dir, short.name) == ("m.pth", "out", "x")
    assert short.tensorboard and short.yes

    long_only = train_cli.parse_args(["-i", "c.yaml", "--restart-training-state", "--log-weights", "--dry-run"])
    assert long_only.restart_training_state and long_only.log_weights and long_only.dry_run


###############################################################################
################################ RUN DIRECTORY ################################
###############################################################################


def test_core_receives_validated_config_and_fresh_run_directory(core, config_file, out_dir):
    run(config_file, out_dir)

    call = core.call
    assert call["save_dir"] == out_dir / "train_unit_000"
    assert call["checkpoint"] is None
    assert isinstance(call["config"], Config)
    assert call["config"].get_str("device") == "cpu", "schema defaults must be applied before training"
    assert call["args"].name == "unit"


def test_run_directory_layout_and_saved_artifacts(core, config_file, out_dir, train_config):
    run(config_file, out_dir)

    save_dir = core.call["save_dir"]
    assert {p.name for p in save_dir.iterdir() if p.is_dir()} == {"models", "checkpoints"}
    for artifact in (
        "train_config.yaml",
        "validated_train_config.yaml",
        "software.info",
        "hardware.info",
        "out.txt",
    ):
        assert (save_dir / artifact).is_file(), artifact
    assert yaml.safe_load((save_dir / "train_config.yaml").read_text()) == train_config, "raw config copied verbatim"
    assert not (save_dir / "merged_train_config.yaml").exists()


def test_validated_config_on_disk_matches_the_one_given_to_core(core, config_file, out_dir):
    run(config_file, out_dir)
    saved = yaml.safe_load((core.call["save_dir"] / "validated_train_config.yaml").read_text())
    assert saved == core.call["config"].as_dict()


def test_run_directory_defaults_to_plain_train_prefix(core, config_file, out_dir):
    train_cli.main(["-i", str(config_file), "-o", str(out_dir)])
    assert core.call["save_dir"].name == "train_000"


@pytest.mark.parametrize("flags, created", [(["--tensorboard", "--log-weights"], True), ([], False)])
def test_tensorboard_directory_follows_the_flag(core, config_file, out_dir, flags, created):
    run(config_file, out_dir, *flags)

    assert (core.call["save_dir"] / "tensorboard").is_dir() is created
    assert core.call["args"].tensorboard is created


###############################################################################
##################################### SEED ####################################
###############################################################################


def test_configured_seed_is_kept(core, write_yaml, train_config, out_dir):
    train_config["seed"] = 1234
    run(write_yaml(train_config), out_dir)

    assert core.call["config"].get_int("seed") == 1234
    saved = yaml.safe_load((core.call["save_dir"] / "validated_train_config.yaml").read_text())
    assert saved["seed"] == 1234


def test_missing_seed_is_replaced_by_a_concrete_one_and_recorded(core, config_file, out_dir):
    run(config_file, out_dir)

    seed = core.call["config"].get_int("seed")
    saved = yaml.safe_load((core.call["save_dir"] / "validated_train_config.yaml").read_text())
    assert saved["seed"] == seed


###############################################################################
#################################### FLAGS ####################################
###############################################################################


@pytest.mark.parametrize("flags, epochs", [(["--dry-run"], 1), ([], 50)])
def test_dry_run_limits_training_to_a_single_epoch(core, write_yaml, train_config, out_dir, flags, epochs):
    train_config["trainer"]["epochs"] = 50
    run(write_yaml(train_config), out_dir, *flags)

    assert core.call["config"].section("trainer").get_int("epochs") == epochs
    assert core.call["args"].dry_run is bool(flags)


@pytest.mark.parametrize("flag, expected", [(["--yes"], True), ([], False)])
def test_yes_flag_enables_auto_confirmation_only_during_the_run(core, config_file, out_dir, flag, expected):
    run(config_file, out_dir, *flag)

    assert core.auto_yes_during_call == [expected]
    assert is_auto_yes_enabled() is False, "the auto-yes scope must end with the run"


###############################################################################
################################ INVALID INPUT ################################
###############################################################################


def test_invalid_config_is_rejected_before_any_directory_is_created(core, write_yaml, train_config, out_dir):
    train_config["model"]["model_type"] = "nope"

    with pytest.raises(ConfigError, match="Invalid train config"):
        run(write_yaml(train_config), out_dir)

    assert core.calls == []
    assert not out_dir.exists()


###############################################################################
############################ CHECKPOINT (-m) HANDLING #########################
###############################################################################


def _resumable_config(train_config) -> dict:
    """A fine-tuning config: dimensions come from the checkpoint signature instead of 'auto'."""
    config = copy.deepcopy(train_config)
    del config["model"]["in_size"], config["model"]["out_size"]
    return config


@pytest.fixture
def save_checkpoint(signature):
    """Return a helper that writes a checkpoint carrying the standard signature (and optional training state)."""

    def _save(path: Path, training_state: dict | None = None) -> Path:
        Checkpoint(Config(signature), [{}], training_state).save(path)
        return path

    return _save


def test_checkpoint_signature_completes_the_config_and_is_forwarded(
    core, write_yaml, train_config, out_dir, tmp_path, save_checkpoint
):
    checkpoint_path = save_checkpoint(tmp_path / "final.pth")

    run(write_yaml(_resumable_config(train_config)), out_dir, "-m", str(checkpoint_path))

    call = core.call
    assert isinstance(call["checkpoint"], Checkpoint)
    assert call["config"].section("model").get_int("in_size") == 6
    assert call["config"].section("model").get_int("out_size") == 10
    assert (call["save_dir"] / "merged_train_config.yaml").is_file()


def test_checkpoint_signature_conflicting_with_the_config_is_rejected(
    core, write_yaml, train_config, out_dir, tmp_path, save_checkpoint
):
    checkpoint_path = save_checkpoint(tmp_path / "final.pth")
    config = _resumable_config(train_config)
    config["model"]["hidden_size"] = 64  # the checkpoint was trained with 8

    with pytest.raises(ConfigError, match="Conflict for key 'hidden_size'"):
        run(write_yaml(config), out_dir, "-m", str(checkpoint_path))
    assert core.calls == []


def _training_state(learning_rate: float) -> dict:
    return {
        "epochs": [3],
        "optimizer_signature": {"trainer": {"learning_rate": learning_rate}},
        "optimizer_states": [{}],
        "scheduler_signature": {"trainer": {"lr_warmup": False}},
        "epoch_scheduler_states": [{}],
        "warmup_scheduler_states": [None],
    }


def test_training_state_signatures_are_merged_when_resuming_from_a_full_checkpoint(
    core, write_yaml, train_config, out_dir, tmp_path, save_checkpoint
):
    save_checkpoint(tmp_path / "final.pth", _training_state(0.001))

    run(write_yaml(_resumable_config(train_config)), out_dir, "-m", str(tmp_path / "final_full.pth"))

    assert core.call["checkpoint"].training_state is not None
    assert core.call["config"].section("trainer").get_bool("lr_warmup") is False


def test_conflicting_optimizer_signature_is_rejected_when_resuming(
    core, write_yaml, train_config, out_dir, tmp_path, save_checkpoint
):
    save_checkpoint(tmp_path / "final.pth", _training_state(learning_rate=0.5))

    with pytest.raises(ConfigError, match="Conflict for key 'learning_rate'"):
        run(write_yaml(_resumable_config(train_config)), out_dir, "-m", str(tmp_path / "final_full.pth"))


def test_restart_training_state_ignores_the_optimizer_and_scheduler_signatures(
    core, write_yaml, train_config, out_dir, tmp_path, save_checkpoint
):
    save_checkpoint(tmp_path / "final.pth", _training_state(learning_rate=0.5))

    run(
        write_yaml(_resumable_config(train_config)),
        out_dir,
        "-m",
        str(tmp_path / "final_full.pth"),
        "--restart-training-state",
    )

    assert core.call["config"].section("trainer").get_float("learning_rate") == 0.001
    assert core.call["args"].restart_training_state is True


###############################################################################
################################ BACKEND DISPATCH #############################
###############################################################################


def test_lightning_trainer_is_dispatched_to_the_lightning_backend(
    monkeypatch, core, write_yaml, train_config, out_dir
):
    launched = Recorder()
    monkeypatch.setattr("xanesnet.core_lightning.train_initial", launched)
    train_config["trainer"] = {"trainer_type": "lightning", "epochs": 1, "batch_size": 2}

    run(write_yaml(train_config), out_dir)

    assert core.calls == []
    assert launched.call["config"].section("trainer").get_str("trainer_type") == "lightning"


def test_prepared_lightning_worker_skips_the_normal_startup(monkeypatch, core, tmp_path, out_dir):
    workers: list[Path] = []
    monkeypatch.setattr("xanesnet.core_lightning.train_worker", workers.append)
    context = tmp_path / "lightning_context.json"
    monkeypatch.setenv(LIGHTNING_CONTEXT_ENV, str(context))

    # The configuration file does not even need to exist: workers load the prepared run context instead.
    train_cli.main(["-i", str(tmp_path / "ignored.yaml"), "-o", str(out_dir)])

    assert workers == [context]
    assert core.calls == []
    assert not out_dir.exists(), "workers must not create run directories"
