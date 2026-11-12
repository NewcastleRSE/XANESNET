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

"""Unit tests for the ``Dataset`` base class: preparing, caching, preloading and splitting processed samples."""

import json
import logging

import pytest
import torch

from xanesnet.datasets.base import TorchDataset
from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError


class TinyDataset(TorchDataset):
    """Source item ``i`` yields ``counts[i]`` samples whose value is ``10 * i + seq``."""

    def __init__(self, tmp_path, counts=(1, 2, 0, 1), preload=False, skip_prepare=False, split_ratios=None,
                 split_indexfile=None, root=None):  # fmt: skip
        super().__init__(
            "tiny",
            datasource=list(range(len(counts))),
            root=str(root or tmp_path / "processed"),
            preload=preload,
            skip_prepare=skip_prepare,
            split_ratios=split_ratios,
            split_indexfile=split_indexfile,
        )
        self.counts = counts
        self.prepared_items: list[int] = []

    def _prepare_single(self, idx, save_path_fn):
        self.prepared_items.append(idx)
        for seq in range(self.counts[idx]):
            torch.save(torch.tensor(10 * idx + seq), save_path_fn(seq))
        return self.counts[idx]

    def _load_item(self, path):
        return torch.load(path)

    @property
    def signature(self):
        return super().signature


def prepared(tmp_path, **kwargs) -> TinyDataset:
    dataset = TinyDataset(tmp_path, **kwargs)
    dataset.prepare()
    return dataset


###############################################################################
################################## PREPARATION ################################
###############################################################################


def test_the_length_is_unknown_before_preparation(tmp_path):
    with pytest.raises(ValueError, match="call prepare"):
        len(TinyDataset(tmp_path))


def test_prepared_samples_are_numbered_consecutively_across_source_items(tmp_path):
    dataset = prepared(tmp_path)

    assert len(dataset) == 4  # item 2 yields nothing and is skipped
    assert [int(dataset[i]) for i in range(4)] == [0, 10, 11, 30]
    assert [path.rsplit("/", 1)[-1] for path in dataset.processed_files] == ["0.pth", "1.pth", "2.pth", "3.pth"]


def test_the_processed_directory_is_created_when_missing(tmp_path):
    dataset = prepared(tmp_path, root=tmp_path / "a" / "b")
    assert (tmp_path / "a" / "b" / "0.pth").exists()
    assert dataset.processed_dir == str(tmp_path / "a" / "b")


def test_a_non_empty_processed_directory_is_cleared_when_the_user_agrees(tmp_path, confirm, caplog):
    prepared(tmp_path, counts=(3,))
    (tmp_path / "processed" / "stale.txt").write_text("old")

    with caplog.at_level(logging.WARNING):
        dataset = prepared(tmp_path, counts=(1,))

    assert confirm.asked == 1
    assert "not empty" in caplog.text
    assert len(dataset) == 1
    assert sorted(p.name for p in (tmp_path / "processed").iterdir()) == ["0.pth"]


def test_a_non_empty_processed_directory_is_kept_when_the_user_declines(tmp_path, confirm):
    prepared(tmp_path, counts=(2,))
    confirm.answer = False

    with pytest.raises(ConfigError, match="not empty"):
        prepared(tmp_path, counts=(1,))

    assert (tmp_path / "processed" / "1.pth").exists()


def test_skip_prepare_reuses_existing_files_without_processing_anything(tmp_path):
    prepared(tmp_path, counts=(2, 1))

    reused = prepared(tmp_path, skip_prepare=True)

    assert reused.prepared_items == []
    assert len(reused) == 3
    assert int(reused[2]) == 10


###############################################################################
#################################### PRELOAD ##################################
###############################################################################


def test_preloading_serves_samples_from_memory(tmp_path):
    dataset = prepared(tmp_path, preload=True)

    assert dataset.check_preload() is True
    for path in dataset.processed_files:
        (tmp_path / "processed" / path.rsplit("/", 1)[-1]).unlink()

    assert [int(dataset[i]) for i in range(4)] == [0, 10, 11, 30]


def test_without_preloading_samples_are_read_from_disk_on_access(tmp_path):
    dataset = prepared(tmp_path, preload=False)

    assert dataset.check_preload() is False
    assert dataset.inmemory_dataset == []
    assert int(dataset[1]) == 10


###############################################################################
#################################### SPLITS ###################################
###############################################################################


def test_ratio_splits_partition_the_samples_with_the_remainder_in_the_last_subset(tmp_path):
    dataset = prepared(tmp_path, counts=(1,) * 10, split_ratios=[0.6, 0.2, 0.2])

    (plan,) = dataset.setup_splits()

    assert [len(indices) for indices in plan] == [6, 2, 2]
    assert sorted(i for indices in plan for i in indices) == list(range(10))
    assert [len(subset) for subset in dataset.subsets] == [6, 2, 2]
    assert dataset.train_subset.indices == plan[0]
    assert dataset.valid_subset.indices == plan[1]


def test_rounding_leftovers_end_up_in_the_last_subset(tmp_path):
    dataset = prepared(tmp_path, counts=(1,) * 7, split_ratios=[0.5, 0.5])

    (plan,) = dataset.setup_splits()

    assert [len(indices) for indices in plan] == [3, 4]


def test_ratios_must_sum_to_one(tmp_path):
    dataset = prepared(tmp_path, split_ratios=[0.5, 0.2])

    with pytest.raises(ConfigError, match="must sum to 1.0"):
        dataset.setup_splits()


def test_splits_can_come_from_an_index_file(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"models": [{"train": [0, 1], "valid": [2, 3]}, {"train": [3], "valid": [0]}]}))
    dataset = prepared(tmp_path, split_indexfile=str(path))

    plans = dataset.setup_splits()

    assert plans == [[[0, 1], [2, 3]], [[3], [0]]]
    assert dataset.has_split_indexfile is True
    assert dataset.get_subset_indices(1) == [2, 3]


def test_without_split_configuration_there_are_no_subsets(tmp_path, caplog):
    dataset = prepared(tmp_path)

    with caplog.at_level(logging.WARNING):
        assert dataset.setup_splits() == []

    assert "No split_ratios or split_indexfile" in caplog.text
    assert dataset.subsets == []
    assert dataset.train_subset is None and dataset.valid_subset is None
    assert dataset.get_subset_indices(0) is None
    assert dataset.has_split_indexfile is False


def test_set_splits_replaces_the_subsets_and_out_of_range_lookups_return_none(tmp_path):
    dataset = prepared(tmp_path)

    dataset.set_splits([[[0, 1, 2], [3]]])

    assert dataset.get_all_subset_indices() == [[0, 1, 2], [3]]
    assert dataset.get_subset_indices(2) is None
    assert dataset.get_subset(-1) is None


def test_a_dataset_view_exposes_one_split_plan_and_leaves_the_original_untouched(tmp_path):
    dataset = prepared(tmp_path)
    dataset.set_splits([[[0, 1], [2]], [[2, 3], [0]]])

    view = dataset.get_dataset(1)

    assert view.get_all_subset_indices() == [[2, 3], [0]]
    assert dataset.get_all_subset_indices() == [[0, 1], [2]]
    assert len(view) == len(dataset)


def test_dataset_views_need_split_plans(tmp_path):
    with pytest.raises(ConfigError, match="before split plans are set"):
        prepared(tmp_path).get_dataset(0)


def test_the_signature_identifies_the_dataset_type(tmp_path):
    assert prepared(tmp_path).signature.as_dict() == {"dataset_type": "tiny"}
    assert isinstance(prepared(tmp_path).signature, Config)


def test_torch_datasets_collate_samples_into_stacked_batches(tmp_path):
    dataset = prepared(tmp_path)

    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert batch.tolist() == [0, 10]
