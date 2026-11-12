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

"""Unit tests for the environment-based Lightning worker context."""

from pathlib import Path

import pytest

from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.lightning import LIGHTNING_CONTEXT_ENV, get_lightning_context_path


def test_absent_variable_means_no_context(monkeypatch):
    monkeypatch.delenv(LIGHTNING_CONTEXT_ENV, raising=False)
    assert get_lightning_context_path() is None


def test_absolute_path_is_returned_as_path(monkeypatch, tmp_path):
    monkeypatch.setenv(LIGHTNING_CONTEXT_ENV, str(tmp_path / "context.pt"))
    assert get_lightning_context_path() == tmp_path / "context.pt"
    assert isinstance(get_lightning_context_path(), Path)


@pytest.mark.parametrize("value", ["", "relative/context.pt", "./context.pt"])
def test_empty_or_relative_values_are_rejected(monkeypatch, value):
    monkeypatch.setenv(LIGHTNING_CONTEXT_ENV, value)
    with pytest.raises(ConfigError, match="absolute context path"):
        get_lightning_context_path()


def test_reading_the_context_does_not_import_lightning(monkeypatch):
    """The helper exists so non-Lightning runs never pay for (or require) the Lightning import."""
    import sys

    monkeypatch.delenv(LIGHTNING_CONTEXT_ENV, raising=False)
    modules_before = {name for name in sys.modules if name.startswith(("lightning", "pytorch_lightning"))}
    get_lightning_context_path()
    modules_after = {name for name in sys.modules if name.startswith(("lightning", "pytorch_lightning"))}
    assert modules_after == modules_before
