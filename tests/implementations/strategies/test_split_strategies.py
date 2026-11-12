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

"""Unit tests for the data splits that ``deep_ensemble``, ``bootstrap``, ``kfold`` create for their members."""

import pytest

from xanesnet.utils.exceptions import ConfigError

from . import fakes


###############################################################################
################################ DEEP ENSEMBLE ################################
###############################################################################


def test_deep_ensemble_members_share_the_single_configured_split(build):
    dataset = fakes.FakeDataset(size=6, plans=[[[0, 1, 2, 3], [4, 5]]])

    splits = build("deep_ensemble", dataset, n_models=3).setup_model_splits()

    assert splits == [[[0, 1, 2, 3], [4, 5]]] * 3
    assert dataset.plans == splits


def test_deep_ensemble_keeps_explicit_per_member_splits(build):
    plans = [[[0, 1], [2]], [[1, 2], [0]]]
    dataset = fakes.FakeDataset(size=3, plans=plans)

    assert build("deep_ensemble", dataset, n_models=2).setup_model_splits() == plans


###############################################################################
################################## BOOTSTRAP ##################################
###############################################################################


def test_bootstrap_resamples_the_training_subset_with_replacement_for_every_member(build):
    dataset = fakes.FakeDataset(size=10, plans=[[list(range(8)), [8, 9]]])

    splits = build("bootstrap", dataset, n_models=4, sample_fraction=1.0).setup_model_splits()

    assert len(splits) == 4
    for train, valid in splits:
        assert len(train) == 8
        assert set(train) <= set(range(8))
        assert valid == [8, 9]  # validation data is never resampled
    assert len({tuple(train) for train, _ in splits}) > 1
    assert dataset.plans == splits


@pytest.mark.parametrize(("fraction", "expected"), [(0.5, 4), (0.25, 2), (0.01, 1)])
def test_bootstrap_sample_size_follows_the_fraction_with_a_minimum_of_one(build, fraction, expected):
    dataset = fakes.FakeDataset(size=10, plans=[[list(range(8)), [8, 9]]])

    splits = build("bootstrap", dataset, n_models=2, sample_fraction=fraction).setup_model_splits()

    assert [len(train) for train, _ in splits] == [expected] * 2


def test_bootstrap_without_a_validation_subset_returns_only_training_samples(build):
    dataset = fakes.FakeDataset(size=5, plans=[[list(range(5))]])

    splits = build("bootstrap", dataset, n_models=2, sample_fraction=1.0).setup_model_splits()

    assert all(len(plan) == 1 for plan in splits)


def test_bootstrap_uses_the_whole_dataset_when_no_split_is_configured(build):
    dataset = fakes.FakeDataset(size=5, plans=[])

    splits = build("bootstrap", dataset, n_models=2, sample_fraction=1.0).setup_model_splits()

    assert all(len(plan[0]) == 5 and set(plan[0]) <= set(range(5)) for plan in splits)


def test_bootstrap_with_an_empty_training_subset_is_rejected(build):
    dataset = fakes.FakeDataset(size=2, plans=[[[], [0, 1]]])

    with pytest.raises(ConfigError, match="non-empty training subset"):
        build("bootstrap", dataset, n_models=2, sample_fraction=1.0).setup_model_splits()


def test_bootstrap_respects_an_explicit_split_index_file(build):
    plans = [[[0, 1], [2]], [[1, 2], [0]]]
    dataset = fakes.FakeDataset(size=3, plans=plans, indexfile=True)

    assert build("bootstrap", dataset, n_models=2, sample_fraction=1.0).setup_model_splits() == plans


###############################################################################
#################################### KFOLD ####################################
###############################################################################


def test_kfold_creates_n_splits_times_n_repeats_members(build):
    strategy = build("kfold", fakes.FakeDataset(size=10), n_splits=3, n_repeats=2)

    splits = strategy.setup_model_splits()

    assert strategy.n_models == 6
    assert len(splits) == 6


@pytest.mark.parametrize(("size", "n_splits"), [(10, 5), (11, 3), (7, 7)])
def test_every_sample_is_validated_exactly_once_per_repeat(build, size, n_splits):
    dataset = fakes.FakeDataset(size=size)

    splits = build("kfold", dataset, n_splits=n_splits, n_repeats=2).setup_model_splits()

    for repeat in (splits[:n_splits], splits[n_splits:]):
        validation = [index for _, valid in repeat for index in valid]
        assert sorted(validation) == list(range(size))
    for train, valid in splits:
        assert sorted(train + valid) == list(range(size))
        assert not set(train) & set(valid)
    assert dataset.plans == splits


def test_repeats_use_different_shuffles(build):
    splits = build("kfold", fakes.FakeDataset(size=30), n_splits=3, n_repeats=2).setup_model_splits()

    first_repeat = [tuple(valid) for _, valid in splits[:3]]
    second_repeat = [tuple(valid) for _, valid in splits[3:]]
    assert first_repeat != second_repeat


def test_kfold_requires_at_least_one_validation_sample_per_fold(build):
    with pytest.raises(ConfigError, match="n_splits=5 exceeds the dataset size 3"):
        build("kfold", fakes.FakeDataset(size=3), n_splits=5, n_repeats=1).setup_model_splits()


def test_kfold_respects_an_explicit_split_index_file(build):
    plans = [[[0, 1], [2]], [[1, 2], [0]]]
    dataset = fakes.FakeDataset(size=3, plans=plans, indexfile=True)

    assert build("kfold", dataset, n_splits=2, n_repeats=1).setup_model_splits() == plans


def test_kfold_derives_encoding_statistics_from_all_samples_without_touching_the_dataset_splits(build):
    dataset = fakes.FakeDataset(size=9)
    strategy = build("kfold", dataset, n_splits=3, n_repeats=1)
    splits = strategy.setup_model_splits()

    encoding_dataset = strategy._dataset_for_encoding()

    assert encoding_dataset.get_subset_indices(0) == list(range(9))
    assert dataset.plans == splits
