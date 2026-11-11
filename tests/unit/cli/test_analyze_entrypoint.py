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

"""Unit tests for the ``xanesnet analyze`` entry point (argument parsing and run orchestration)."""

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from xanesnet import analyze as analyze_cli
from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.prompts import is_auto_yes_enabled


class Recorder:
    """Captures what the entry point passes to the core ``analyze`` function."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.auto_yes_during_call: list[bool] = []

    def __call__(self, config: Config, args: Namespace, save_dir: Path) -> None:
        self.calls.append({"config": config, "args": args, "save_dir": save_dir})
        self.auto_yes_during_call.append(is_auto_yes_enabled())

    @property
    def call(self) -> dict:
        assert len(self.calls) == 1, f"expected exactly one core call, got {len(self.calls)}"
        return self.calls[0]


@pytest.fixture
def core(monkeypatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(analyze_cli, "analyze", recorder)
    return recorder


@pytest.fixture
def config_file(write_yaml, analyze_config) -> Path:
    return write_yaml(analyze_config, name="analyze.yaml")


@pytest.fixture
def out_dir(tmp_path) -> Path:
    return tmp_path / "runs"


def run(config_file: Path, out_dir: Path, *extra: str) -> None:
    analyze_cli.main(["-i", str(config_file), "-r", "run_a", "-o", str(out_dir), "-n", "unit", *extra])


###############################################################################
############################### ARGUMENT PARSING ##############################
###############################################################################


def test_optional_arguments_default_to_empty_values():
    args = analyze_cli.parse_args(["-i", "c.yaml", "-r", "run_a"])
    assert args.inference_runs == ["run_a"]
    assert args.prediction_names == []
    assert args.out_dir is None and args.name is None and args.yes is False


@pytest.mark.parametrize(
    "argv",
    [
        ["-r", "a", "b", "c"],
        ["-r", "a", "-r", "b", "-r", "c"],
        ["-r", "a", "b", "-r", "c"],
        ["--inference-runs", "a", "b", "c"],
    ],
    ids=["space-separated", "repeated", "mixed", "long-form"],
)
def test_inference_runs_may_be_repeated_and_or_space_separated(argv):
    args = analyze_cli.parse_args(["-i", "c.yaml", *argv])
    assert args.inference_runs == ["a", "b", "c"]


@pytest.mark.parametrize(
    "argv",
    [["-d", "X", "Y"], ["-d", "X", "-d", "Y"], ["--prediction-names", "X", "Y"]],
    ids=["space-separated", "repeated", "long-form"],
)
def test_prediction_names_are_flattened_in_order(argv):
    args = analyze_cli.parse_args(["-i", "c.yaml", "-r", "a", "b", *argv])
    assert args.prediction_names == ["X", "Y"]


@pytest.mark.parametrize("missing", ["-i", "-r"])
def test_input_file_and_inference_runs_are_required(missing):
    argv = ["-i", "c.yaml", "-r", "a"]
    index = argv.index(missing)
    del argv[index : index + 2]
    with pytest.raises(SystemExit) as excinfo:
        analyze_cli.parse_args(argv)
    assert excinfo.value.code == 2


def test_inference_runs_require_at_least_one_value():
    with pytest.raises(SystemExit):
        analyze_cli.parse_args(["-i", "c.yaml", "-r"])


###############################################################################
################################ RUN DIRECTORY ################################
###############################################################################


def test_core_receives_validated_config_and_parsed_arguments(core, config_file, out_dir):
    analyze_cli.main(
        ["-i", str(config_file), "-r", "a", "b", "-d", "A", "B", "-o", str(out_dir), "-n", "unit"]
    )

    call = core.call
    assert call["save_dir"] == out_dir / "analyze_unit_000"
    assert call["args"].inference_runs == ["a", "b"]
    assert call["args"].prediction_names == ["A", "B"]
    assert call["config"].get_bool("preload") is False, "schema defaults must be applied"


def test_run_directory_layout_and_saved_artifacts(core, config_file, out_dir, analyze_config):
    run(config_file, out_dir)

    save_dir = core.call["save_dir"]
    assert {p.name for p in save_dir.iterdir() if p.is_dir()} == {"plots", "reports", "aux"}
    for artifact in ("analyze_config.yaml", "validated_analyze_config.yaml", "software.info", "hardware.info", "out.txt"):
        assert (save_dir / artifact).is_file(), artifact
    assert yaml.safe_load((save_dir / "analyze_config.yaml").read_text()) == analyze_config


def test_consecutive_runs_get_consecutive_directories(core, config_file, out_dir):
    run(config_file, out_dir)
    run(config_file, out_dir)
    assert [c["save_dir"].name for c in core.calls] == ["analyze_unit_000", "analyze_unit_001"]


def test_configured_seed_is_applied(core, write_yaml, analyze_config, out_dir):
    analyze_config["seed"] = 5
    run(write_yaml(analyze_config), out_dir)
    assert core.call["config"].get_int("seed") == 5


@pytest.mark.parametrize("flag, expected", [(["--yes"], True), ([], False)])
def test_yes_flag_enables_auto_confirmation_only_during_the_run(core, config_file, out_dir, flag, expected):
    run(config_file, out_dir, *flag)
    assert core.auto_yes_during_call == [expected]
    assert is_auto_yes_enabled() is False


###############################################################################
################################ INVALID INPUT ################################
###############################################################################


def test_missing_config_file_is_reported(core, tmp_path, out_dir):
    with pytest.raises(ConfigError, match="does not exist"):
        run(tmp_path / "missing.yaml", out_dir)
    assert core.calls == []


def test_invalid_config_is_rejected(core, write_yaml, analyze_config, out_dir):
    analyze_config["selectors"] = [{"selector_type": "nope"}]

    with pytest.raises(ConfigError, match=r"Invalid analyze config at config\.selectors"):
        run(write_yaml(analyze_config), out_dir)
    assert core.calls == []
