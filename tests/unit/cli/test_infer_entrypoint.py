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

"""Unit tests for the ``xanesnet infer`` entry point (argument parsing and run orchestration)."""

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from xanesnet import infer as infer_cli
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.prompts import is_auto_yes_enabled


class Recorder:
    """Captures what the entry point passes to the core ``infer`` function."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.auto_yes_during_call: list[bool] = []

    def __call__(self, config: Config, args: Namespace, save_dir: Path, checkpoint: Checkpoint) -> None:
        self.calls.append({"config": config, "args": args, "save_dir": save_dir, "checkpoint": checkpoint})
        self.auto_yes_during_call.append(is_auto_yes_enabled())

    @property
    def call(self) -> dict:
        assert len(self.calls) == 1, f"expected exactly one core call, got {len(self.calls)}"
        return self.calls[0]


@pytest.fixture
def core(monkeypatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(infer_cli, "infer", recorder)
    return recorder


@pytest.fixture
def config_file(write_yaml, infer_user_config) -> Path:
    return write_yaml(infer_user_config, name="infer.yaml")


@pytest.fixture
def checkpoint_file(tmp_path, signature) -> Path:
    path = tmp_path / "final.pth"
    Checkpoint(Config(signature), [{}]).save(path)
    return path


@pytest.fixture
def out_dir(tmp_path) -> Path:
    return tmp_path / "runs"


def run(config_file: Path, checkpoint_file: Path, out_dir: Path, *extra: str) -> None:
    infer_cli.main(["-i", str(config_file), "-m", str(checkpoint_file), "-o", str(out_dir), "-n", "unit", *extra])


###############################################################################
############################### ARGUMENT PARSING ##############################
###############################################################################


def test_defaults_for_optional_arguments():
    args = infer_cli.parse_args(["-i", "c.yaml", "-m", "m.pth"])
    assert (args.in_file, args.in_model) == ("c.yaml", "m.pth")
    assert args.out_dir is None and args.name is None and args.yes is False

    long = infer_cli.parse_args(
        ["--in_file", "c.yaml", "--in_model", "m.pth", "--out_dir", "out", "--name", "x", "--yes"]
    )
    assert vars(long) == vars(infer_cli.parse_args(["-i", "c.yaml", "-m", "m.pth", "-o", "out", "-n", "x", "-y"]))


###############################################################################
################################ RUN DIRECTORY ################################
###############################################################################


def test_core_receives_validated_merged_config_and_loaded_checkpoint(core, config_file, checkpoint_file, out_dir):
    run(config_file, checkpoint_file, out_dir)

    call = core.call
    assert call["save_dir"] == out_dir / "infer_unit_000"
    assert isinstance(call["checkpoint"], Checkpoint)
    config = call["config"]
    assert config.section("model").get_str("model_type") == "mlp", "model section comes from the checkpoint signature"
    assert config.get_str("device") == "cpu", "schema defaults must be applied"
    assert config.section("inferencer").get_bool("shuffle") is False


def test_run_directory_layout_and_saved_artifacts(core, config_file, checkpoint_file, out_dir, infer_user_config):
    run(config_file, checkpoint_file, out_dir)

    save_dir = core.call["save_dir"]
    assert (save_dir / "predictions").is_dir()
    for artifact in (
        "infer_config.yaml",
        "merged_infer_config.yaml",
        "validated_infer_config.yaml",
        "software.info",
        "hardware.info",
        "out.txt",
    ):
        assert (save_dir / artifact).is_file(), artifact
    assert yaml.safe_load((save_dir / "infer_config.yaml").read_text()) == infer_user_config


def test_merged_config_combines_user_config_and_signature(core, config_file, checkpoint_file, out_dir, signature):
    run(config_file, checkpoint_file, out_dir)

    merged = yaml.safe_load((core.call["save_dir"] / "merged_infer_config.yaml").read_text())
    assert merged["model"] == signature["model"]
    assert merged["encodings"] == signature["encodings"]
    assert merged["inferencer"]["inferencer_type"] == "basic"


def test_run_directory_defaults_to_plain_infer_prefix(core, config_file, checkpoint_file, out_dir):
    infer_cli.main(["-i", str(config_file), "-m", str(checkpoint_file), "-o", str(out_dir)])
    assert core.call["save_dir"].name == "infer_000"


###############################################################################
##################################### SEED ####################################
###############################################################################


@pytest.mark.parametrize("flag, expected", [(["--yes"], True), ([], False)])
def test_yes_flag_enables_auto_confirmation_only_during_the_run(
    core, config_file, checkpoint_file, out_dir, flag, expected
):
    run(config_file, checkpoint_file, out_dir, *flag)
    assert core.auto_yes_during_call == [expected]
    assert is_auto_yes_enabled() is False


###############################################################################
################################ INVALID INPUT ################################
###############################################################################


def test_user_config_conflicting_with_the_signature_is_rejected(
    core, write_yaml, infer_user_config, checkpoint_file, out_dir
):
    infer_user_config["model"] = {"model_type": "mlp", "hidden_size": 64}  # checkpoint was trained with 8

    with pytest.raises(ConfigError, match="Conflict for key 'hidden_size'"):
        run(write_yaml(infer_user_config), checkpoint_file, out_dir)
    assert core.calls == []


def test_invalid_user_config_is_rejected_after_merging(core, write_yaml, infer_user_config, checkpoint_file, out_dir):
    infer_user_config["inferencer"]["inferencer_type"] = "nope"

    with pytest.raises(ConfigError, match=r"Invalid infer config at config\.inferencer"):
        run(write_yaml(infer_user_config), checkpoint_file, out_dir)
    assert core.calls == []
