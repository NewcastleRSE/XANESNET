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

"""Unit tests for the helpers of the multiprocessing dataset mixin (the process pool itself is not started)."""

import os

import pytest

from xanesnet.datasets import DatasetRegistry
from xanesnet.datasets._mp import MpDatasetMixin


@pytest.mark.parametrize(
    ("total", "chunks", "expected"),
    [
        (10, 3, [(0, 4), (4, 7), (7, 10)]),
        (9, 3, [(0, 3), (3, 6), (6, 9)]),
        (2, 5, [(0, 1), (1, 2)]),
        (5, 1, [(0, 5)]),
        (5, 0, [(0, 5)]),
    ],
)
def test_samples_are_split_into_contiguous_balanced_ranges(total, chunks, expected):
    assert MpDatasetMixin._split_index_ranges(total, chunks) == expected


@pytest.mark.parametrize("total", [0, -1])
def test_nothing_is_split_for_an_empty_source(total):
    assert MpDatasetMixin._split_index_ranges(total, 4) == []


def test_ranges_cover_every_sample_exactly_once():
    ranges = MpDatasetMixin._split_index_ranges(103, 8)

    covered = [i for start, end in ranges for i in range(start, end)]
    assert covered == list(range(103))
    assert max(end - start for start, end in ranges) - min(end - start for start, end in ranges) <= 1


@pytest.mark.parametrize(("requested", "expected"), [(4, 4), (1, 1)])
def test_an_explicit_worker_count_is_used_as_given(requested, expected):
    assert MpDatasetMixin._resolve_num_workers(requested) == expected


@pytest.mark.parametrize("requested", [None, 0, -2])
def test_a_missing_or_non_positive_worker_count_means_every_cpu(requested):
    assert MpDatasetMixin._resolve_num_workers(requested) == (os.cpu_count() or 1)


def test_worker_output_names_sort_in_sample_order():
    paths = [MpDatasetMixin._mp_save_path("out", idx, seq) for idx, seq in [(10, 0), (2, 1), (2, 0), (1, 5)]]

    assert sorted(paths) == [
        os.path.join("out", "0000000001_000005.pth"),
        os.path.join("out", "0000000002_000000.pth"),
        os.path.join("out", "0000000002_000001.pth"),
        os.path.join("out", "0000000010_000000.pth"),
    ]


def test_every_message_passing_dataset_has_a_plain_counterpart_with_the_same_prepare_logic():
    mp_keys = [key for key in DatasetRegistry.list() if key.endswith("_mp")]

    assert mp_keys
    for key in mp_keys:
        mp_class = DatasetRegistry.get(key)
        plain_class = DatasetRegistry.get(key.removesuffix("_mp"))
        assert issubclass(mp_class, MpDatasetMixin)
        assert issubclass(mp_class, plain_class)
        assert not issubclass(plain_class, MpDatasetMixin)
