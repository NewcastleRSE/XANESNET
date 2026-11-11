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

"""Unit tests for ``JSONLStream`` and ``json_friendly``."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from xanesnet.serialization.jsonl_stream import JSONLStream, json_friendly

RECORDS = [{"id": 0, "x": [1, 2]}, {"id": 1, "x": []}, {"id": 2, "x": [3]}]


@pytest.fixture
def jsonl_file(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    return path


###############################################################################
################################## JSONLStream ################################
###############################################################################


def test_iteration_yields_records_in_file_order(jsonl_file):
    assert list(JSONLStream(jsonl_file)) == RECORDS


def test_iteration_is_repeatable(jsonl_file):
    stream = JSONLStream(jsonl_file)
    assert list(stream) == list(stream)


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n')
    assert list(JSONLStream(path)) == [{"a": 1}, {"a": 2}]


def test_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert list(JSONLStream(path)) == []
    assert len(JSONLStream(path)) == 0


def test_explicit_count_is_trusted_without_touching_the_file(tmp_path):
    assert len(JSONLStream(tmp_path / "does_not_exist.jsonl", count=7)) == 7


def test_count_is_read_from_the_meta_sidecar(jsonl_file):
    jsonl_file.with_suffix(".meta.json").write_text(json.dumps({"count": 42}))
    assert len(JSONLStream(jsonl_file)) == 42


def test_count_falls_back_to_scanning_and_warns(jsonl_file, caplog):
    with caplog.at_level(logging.WARNING):
        assert len(JSONLStream(jsonl_file)) == len(RECORDS)
    assert "Counting lines" in caplog.text


def test_scanned_count_ignores_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n\n')
    assert len(JSONLStream(path)) == 2


def test_count_is_cached_after_the_first_lookup(jsonl_file):
    stream = JSONLStream(jsonl_file)
    assert len(stream) == 3
    jsonl_file.write_text("")
    assert len(stream) == 3


###############################################################################
################################# json_friendly ###############################
###############################################################################


@pytest.mark.parametrize("value", [None, "text", 3, 2.5, True])
def test_json_primitives_pass_through(value):
    assert json_friendly(value) == value


def test_paths_become_strings():
    assert json_friendly(Path("/tmp/x")) == "/tmp/x"


def test_containers_are_converted_recursively_and_tuples_become_lists():
    value = {"path": Path("a"), "items": (1, np.float32(2.0), [np.int64(3)])}
    assert json_friendly(value) == {"path": "a", "items": [1, 2.0, [3]]}


@pytest.mark.parametrize("scalar, expected_type", [(np.float32(1.5), float), (np.int64(4), int), (np.bool_(True), bool)])
def test_numpy_scalars_are_unwrapped_to_python_types(scalar, expected_type):
    out = json_friendly(scalar)
    assert type(out) is expected_type
    assert out == scalar


def test_arrays_and_tensors_become_nested_lists():
    assert json_friendly(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    assert json_friendly(torch.tensor([[1.0, 2.0]])) == [[1.0, 2.0]]


def test_unknown_objects_fall_back_to_str():
    class Thing:
        def __str__(self) -> str:
            return "thing"

    assert json_friendly(Thing()) == "thing"


def test_result_is_always_json_serializable():
    payload = {"a": np.arange(3), "b": Path("p"), "c": {1, 2}, "d": torch.tensor(2)}
    json.dumps(json_friendly(payload))
