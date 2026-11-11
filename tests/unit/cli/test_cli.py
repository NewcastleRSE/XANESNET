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

"""Unit tests for the top-level ``xanesnet`` command dispatcher."""

import pytest

from xanesnet import cli
from xanesnet.utils.exceptions import ConfigError


@pytest.fixture
def entry_points(monkeypatch):
    """Replace the three workflow entry points with recorders and return the call log."""
    calls: list[tuple[str, list[str]]] = []
    for name in ("train", "infer", "analyze"):
        monkeypatch.setattr(f"xanesnet.{name}.main", lambda args, _name=name: calls.append((_name, list(args))))
    return calls


###############################################################################
################################### HELP ######################################
###############################################################################


@pytest.mark.parametrize("args", [[], ["-h"], ["--help"], ["--help", "train"]])
def test_help_is_printed_and_exits_cleanly(args, entry_points, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(args)

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "Usage:" in output
    for command in ("train", "infer", "analyze"):
        assert command in output
    assert entry_points == []


###############################################################################
################################## DISPATCH ###################################
###############################################################################


@pytest.mark.parametrize("command", ["train", "infer", "analyze"])
def test_command_is_forwarded_with_its_remaining_arguments(command, entry_points):
    cli.main([command, "-i", "config.yaml", "--yes"])
    assert entry_points == [(command, ["-i", "config.yaml", "--yes"])]


def test_only_the_selected_command_runs(entry_points):
    cli.main(["infer", "-i", "x"])
    assert [name for name, _ in entry_points] == ["infer"]


def test_command_arguments_may_be_empty(entry_points):
    cli.main(["train"])
    assert entry_points == [("train", [])]


@pytest.mark.parametrize("command", ["trainn", "TRAIN", "-i", "predict"])
def test_unknown_commands_are_rejected(command, entry_points):
    with pytest.raises(ConfigError, match=f"Incorrect mode: {command}"):
        cli.main([command])
    assert entry_points == []


def test_console_main_uses_the_process_arguments(monkeypatch, entry_points):
    monkeypatch.setattr("sys.argv", ["xanesnet", "analyze", "-i", "c.yaml"])
    cli.console_main()
    assert entry_points == [("analyze", ["-i", "c.yaml"])]
