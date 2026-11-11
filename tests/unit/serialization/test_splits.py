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

"""Unit tests for reading and writing dataset split indices."""

import json

import pytest

from xanesnet.serialization.splits import load_splits, save_separate_splits, save_splits
from xanesnet.utils.exceptions import ResourceError


def _write_json(path, payload):
    path.write_text(json.dumps(payload))
    return path


###############################################################################
################################## ROUND TRIP #################################
###############################################################################


def test_save_then_load_round_trip(tmp_path):
    plans = [[[0, 1, 2], [3]], [[4, 5], [6, 7]]]
    save_splits(tmp_path / "splits.json", plans)
    assert load_splits(tmp_path / "splits.json") == plans


def test_save_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "splits.json"
    save_splits(path, [[[1]]])
    assert path.is_file()


def test_file_layout_is_model_first_with_named_splits(tmp_path):
    save_splits(tmp_path / "splits.json", [[[1, 2], [3]], [[4]]])
    assert json.loads((tmp_path / "splits.json").read_text()) == {
        "models": [{"train": [1, 2], "valid": [3]}, {"train": [4]}]
    }


def test_train_only_and_empty_plans_survive_round_trip(tmp_path):
    save_splits(tmp_path / "splits.json", [[[0, 1]], []])
    assert load_splits(tmp_path / "splits.json") == [[[0, 1]], []]


def test_save_accepts_string_paths(tmp_path):
    save_splits(str(tmp_path / "s.json"), [[[1]]])
    assert load_splits(str(tmp_path / "s.json")) == [[[1]]]


def test_save_rejects_more_than_train_and_valid(tmp_path):
    with pytest.raises(ValueError, match="at most train and validation"):
        save_splits(tmp_path / "splits.json", [[[1], [2], [3]]])
    assert not (tmp_path / "splits.json").exists()


###############################################################################
################################ MODEL SELECTION ##############################
###############################################################################


@pytest.fixture
def three_models(tmp_path):
    plans = [[[0], [1]], [[2], [3]], [[4], [5]]]
    save_splits(tmp_path / "splits.json", plans)
    return tmp_path / "splits.json", plans


@pytest.mark.parametrize("index", [0, 1, 2])
def test_model_index_selects_exactly_one_plan(three_models, index):
    path, plans = three_models
    assert load_splits(path, model_index=index) == [plans[index]]


@pytest.mark.parametrize("index", [-1, 3, 100, True, 1.0])
def test_invalid_model_index_is_rejected(three_models, index):
    path, _ = three_models
    with pytest.raises(ResourceError, match="out of range"):
        load_splits(path, model_index=index)


###############################################################################
################################## VALIDATION #################################
###############################################################################


@pytest.mark.parametrize("payload", [[], {"models": "x"}, {"other": []}, "text", None], ids=repr)
def test_malformed_top_level_structure_is_rejected(tmp_path, payload):
    with pytest.raises(ResourceError, match="Invalid model splits file format"):
        load_splits(_write_json(tmp_path / "s.json", payload))


def test_non_dict_model_entry_is_rejected(tmp_path):
    with pytest.raises(ResourceError, match="Model split entry 0 must be a dict"):
        load_splits(_write_json(tmp_path / "s.json", {"models": [[1, 2]]}))


def test_unexpected_split_names_are_rejected(tmp_path):
    with pytest.raises(ResourceError, match="Unexpected split keys"):
        load_splits(_write_json(tmp_path / "s.json", {"models": [{"train": [1], "test": [2]}]}))


def test_validation_without_training_split_is_rejected(tmp_path):
    with pytest.raises(ResourceError, match="missing 'train'"):
        load_splits(_write_json(tmp_path / "s.json", {"models": [{"valid": [1]}]}))


@pytest.mark.parametrize("indices", ["1,2", [1, "2"], [1.5], [True], [[1]], None], ids=repr)
def test_non_integer_indices_are_rejected(tmp_path, indices):
    with pytest.raises(ResourceError, match="Invalid indices for split 'train'"):
        load_splits(_write_json(tmp_path / "s.json", {"models": [{"train": indices}]}))


###############################################################################
############################## SEPARATE SPLIT FILES ###########################
###############################################################################


def test_separate_files_are_written_per_model_for_multi_model_runs(tmp_path):
    plans = [[[0], [1]], [[2], [3]]]
    save_separate_splits(tmp_path, plans)

    assert load_splits(tmp_path / "split_indices_0.json") == [plans[0]]
    assert load_splits(tmp_path / "split_indices_1.json") == [plans[1]]


def test_separate_files_are_skipped_for_single_model_runs(tmp_path):
    save_separate_splits(tmp_path, [[[0], [1]]])
    assert list(tmp_path.iterdir()) == []


def test_separate_file_can_be_used_to_load_a_single_member(tmp_path):
    save_separate_splits(tmp_path, [[[0]], [[1]], [[2]]])
    assert load_splits(tmp_path / "split_indices_2.json", model_index=0) == [[[2]]]
