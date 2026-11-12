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

"""Unit tests for confirmation prompts and the auto-yes switch."""

import pytest

from xanesnet.utils import prompts
from xanesnet.utils.prompts import auto_yes, confirm_yes_no, is_auto_yes_enabled, set_auto_yes


@pytest.fixture(autouse=True)
def _manual_mode():
    """Start every test with auto-yes disabled (global state is restored by the root conftest)."""
    set_auto_yes(False)


def _answer(monkeypatch, response: str) -> list[str]:
    """Make ``input`` return ``response`` and record the prompts that were shown."""
    shown: list[str] = []

    def fake_input(prompt: str = "") -> str:
        shown.append(prompt)
        return response

    monkeypatch.setattr("builtins.input", fake_input)
    return shown


###############################################################################
################################# AUTO-YES MODE ###############################
###############################################################################


def test_auto_yes_is_disabled_by_default():
    assert is_auto_yes_enabled() is False


def test_set_auto_yes_toggles_the_global_mode():
    set_auto_yes(True)
    assert is_auto_yes_enabled() is True
    set_auto_yes(False)
    assert is_auto_yes_enabled() is False


def test_context_manager_enables_auto_yes_only_inside_the_block():
    with auto_yes():
        assert is_auto_yes_enabled() is True
    assert is_auto_yes_enabled() is False


def test_context_manager_can_override_a_global_enable_with_false():
    set_auto_yes(True)
    with auto_yes(False):
        assert is_auto_yes_enabled() is False
    assert is_auto_yes_enabled() is True


def test_nested_contexts_restore_the_outer_value():
    with auto_yes(True):
        with auto_yes(False):
            assert is_auto_yes_enabled() is False
        assert is_auto_yes_enabled() is True


def test_context_is_restored_when_the_block_raises():
    with pytest.raises(RuntimeError):
        with auto_yes():
            raise RuntimeError("boom")
    assert is_auto_yes_enabled() is False


###############################################################################
################################ confirm_yes_no ###############################
###############################################################################


def test_auto_yes_accepts_without_asking(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("input() must not be called in auto-yes mode"))
    with auto_yes():
        assert confirm_yes_no("Proceed?", default_yes=False) is True


@pytest.mark.parametrize("response", ["y", "Y", "yes", "YES", "  yes  "])
def test_affirmative_answers_return_true(monkeypatch, response):
    _answer(monkeypatch, response)
    assert confirm_yes_no("Proceed?") is True


@pytest.mark.parametrize("response", ["n", "no", "nope", "maybe"])
def test_other_answers_return_false(monkeypatch, response):
    _answer(monkeypatch, response)
    assert confirm_yes_no("Proceed?") is False


@pytest.mark.parametrize("default_yes", [True, False])
def test_empty_answer_selects_the_default(monkeypatch, default_yes):
    _answer(monkeypatch, "")
    assert confirm_yes_no("Proceed?", default_yes=default_yes) is default_yes


@pytest.mark.parametrize("default_yes, suffix", [(True, "[Y/n]"), (False, "[y/N]")])
def test_prompt_shows_the_default_in_its_suffix(monkeypatch, default_yes, suffix):
    shown = _answer(monkeypatch, "y")
    confirm_yes_no("Overwrite?", default_yes=default_yes)
    assert shown == [f"Overwrite? {suffix}: "]


def test_module_exposes_no_interactive_state_between_calls(monkeypatch):
    """A previous ``auto_yes`` context must not leak into later prompts."""
    with auto_yes():
        confirm_yes_no("first")
    _answer(monkeypatch, "n")
    assert confirm_yes_no("second") is False
    assert prompts.is_auto_yes_enabled() is False
