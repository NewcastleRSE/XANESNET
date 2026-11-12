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

"""Unit tests for the ``descriptor`` / ``descriptor_multihead`` datasets (and their inverse variants)."""

import pytest
import torch

from xanesnet.datasets import DatasetRegistry
from xanesnet.datasets.torch.descriptor import DescriptorData
from xanesnet.datasets.torch.descriptor_multihead import DescriptorMultiheadData
from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError

from .conftest import ListSource, spectrum, water

RDC = Config({"descriptor_type": "rdc", "r_min": 0.0, "r_max": 4.0, "dr": 1.0, "alpha": 2.0, "use_charge": False, "use_spin": False})
RDC_WIDTH = 5


def create(key: str, tmp_path, molecules, descriptors=(RDC,), **overrides):
    kwargs = {
        "dataset_type": key,
        "datasource": ListSource(molecules),
        "root": str(tmp_path / "processed"),
        "preload": False,
        "skip_prepare": False,
        "split_ratios": None,
        "split_indexfile": None,
        "descriptors": list(descriptors),
    }
    kwargs.update(overrides)
    dataset = DatasetRegistry.create(key, **kwargs)
    dataset.prepare()
    return dataset


###############################################################################
################################### DESCRIPTOR ################################
###############################################################################


def test_every_site_with_a_spectrum_becomes_one_sample(tmp_path):
    molecules = [
        water("a", [spectrum(), None, spectrum(2.0)]),
        water("b", [None, spectrum(3.0), None]),
    ]

    dataset = create("descriptor", tmp_path, molecules)

    assert len(dataset) == 3
    assert [dataset[i].sample_id for i in range(3)] == ["a", "a", "b"]
    assert [dataset[i].target_site_index.item() for i in range(3)] == [0, 2, 1]
    assert [dataset[i].element.item() for i in range(3)] == [8, 1, 1]


def test_forward_samples_map_descriptor_features_to_the_spectrum(tmp_path):
    dataset = create("descriptor", tmp_path, [water("a", [spectrum(), None, None])])

    sample = dataset[0]

    assert sample.x.shape == (RDC_WIDTH,) and sample.x.dtype == torch.float32
    torch.testing.assert_close(sample.y, torch.tensor([0.1, 0.5, 0.2]))
    torch.testing.assert_close(sample.energies, torch.tensor([1.0, 2.0, 3.0]))


def test_inverse_samples_swap_the_roles_of_features_and_spectrum(tmp_path):
    forward = create("descriptor", tmp_path / "f", [water("a", [spectrum(), None, None])])[0]
    inverse = create("descriptor_inverse", tmp_path / "i", [water("a", [spectrum(), None, None])])[0]

    torch.testing.assert_close(inverse.x, forward.y)
    torch.testing.assert_close(inverse.y, forward.x)


def test_features_of_several_descriptors_are_concatenated(tmp_path):
    narrow = Config({**RDC.as_dict(), "r_max": 2.0})
    molecule = water("a", [spectrum(), None, None])

    single = create("descriptor", tmp_path / "single", [molecule], descriptors=[RDC])[0].x
    combined = create("descriptor", tmp_path / "combined", [molecule], descriptors=[RDC, narrow])[0].x

    assert combined.shape == (RDC_WIDTH + 3,)
    torch.testing.assert_close(combined[:RDC_WIDTH], single)


def test_samples_without_any_spectrum_are_skipped_with_a_warning(tmp_path, caplog):
    molecules = [water("no_spectrum", None), water("ok", [spectrum(), None, None])]

    with caplog.at_level("WARNING"):
        dataset = create("descriptor", tmp_path, molecules)

    assert len(dataset) == 1
    assert dataset[0].sample_id == "ok"
    assert "No spectrum found for sample 0 (no_spectrum)" in caplog.text


def test_batches_stack_the_samples_and_keep_the_sample_ids_as_a_list(tmp_path):
    dataset = create("descriptor", tmp_path, [water("a", [spectrum(), spectrum(2.0), None]), water("b", [spectrum(), None, None])])

    batch = dataset.collate_fn([dataset[0], dataset[1], dataset[2]])

    assert batch.x.shape == (3, RDC_WIDTH)
    assert batch.y.shape == (3, 3)
    assert batch.energies.shape == (3, 3)
    assert batch.element.tolist() == [8, 1, 8]
    assert batch.sample_id == ["a", "a", "b"]


def test_the_signature_records_the_descriptor_configuration(tmp_path):
    dataset = create("descriptor", tmp_path, [water("a", [spectrum(), None, None])])

    signature = dataset.signature.as_dict()

    assert signature["dataset_type"] == "descriptor"
    assert signature["descriptors"] == [RDC.as_dict()]


def test_descriptor_data_survives_a_round_trip_through_disk(tmp_path):
    data = DescriptorData(x=torch.ones(2), y=torch.zeros(3), energies=torch.arange(3.0), sample_id="s",
                          element=torch.tensor(8), target_site_index=torch.tensor(1))  # fmt: skip

    loaded = DescriptorData.load(data.save(str(tmp_path / "sample.pth")))

    assert loaded.sample_id == "s"
    for name in ("x", "y", "energies", "element", "target_site_index"):
        assert torch.equal(getattr(loaded, name), getattr(data, name))


def test_descriptor_data_moves_every_tensor_and_skips_missing_ones():
    data = DescriptorData(x=torch.ones(2))

    assert data.to("cpu") is data
    assert data.y is None


###############################################################################
############################### DESCRIPTOR MULTIHEAD ##########################
###############################################################################


def test_multihead_samples_carry_the_head_of_their_subdirectory(tmp_path):
    molecules = [
        water("a", [spectrum(), None, None], subdir_id=0),
        water("b", [spectrum(), None, None], subdir_id=1),
        water("c", [spectrum(), None, None], subdir_id=1),
    ]

    dataset = create("descriptor_multihead", tmp_path, molecules)

    assert [dataset[i].head_idx.item() for i in range(3)] == [0, 1, 1]
    assert dataset.num_heads == 2


def test_multihead_batches_stack_the_head_indices(tmp_path):
    dataset = create("descriptor_multihead", tmp_path, [water("a", [spectrum(), None, None], subdir_id=0),
                                                         water("b", [spectrum(), None, None], subdir_id=1)])  # fmt: skip

    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert batch.head_idx.tolist() == [0, 1]
    assert batch.x.shape == (2, RDC_WIDTH)


def test_multihead_inverse_swaps_features_and_spectrum(tmp_path):
    molecule = water("a", [spectrum(), None, None], subdir_id=0)
    forward = create("descriptor_multihead", tmp_path / "f", [molecule])[0]
    inverse = create("descriptor_multihead_inverse", tmp_path / "i", [molecule])[0]

    torch.testing.assert_close(inverse.x, forward.y)
    torch.testing.assert_close(inverse.y, forward.x)


def test_multihead_entries_need_a_subdirectory_id(tmp_path):
    with pytest.raises(ConfigError, match="subdir_id"):
        create("descriptor_multihead", tmp_path, [water("a", [spectrum(), None, None])])


def test_head_indices_must_be_contiguous_and_start_at_zero(tmp_path):
    molecules = [water("a", [spectrum(), None, None], subdir_id=0), water("b", [spectrum(), None, None], subdir_id=2)]
    dataset = create("descriptor_multihead", tmp_path, molecules)

    with pytest.raises(ConfigError, match="contiguous zero-based"):
        dataset.num_heads


def test_the_head_count_cannot_be_derived_from_an_empty_dataset(tmp_path):
    dataset = create("descriptor_multihead", tmp_path, [water("a", None, subdir_id=0)])

    with pytest.raises(ConfigError, match="empty dataset"):
        dataset.num_heads


def test_multihead_samples_without_spectra_are_skipped(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        dataset = create("descriptor_multihead", tmp_path, [water("a", None, subdir_id=0), water("b", [spectrum(), None, None], subdir_id=0)])

    assert len(dataset) == 1
    assert "No spectrum found" in caplog.text


def test_multihead_signature_records_the_descriptors(tmp_path):
    dataset = create("descriptor_multihead", tmp_path, [water("a", [spectrum(), None, None], subdir_id=0)])
    assert dataset.signature.as_dict()["descriptors"] == [RDC.as_dict()]


def test_multihead_data_round_trips_through_disk_and_tolerates_missing_fields(tmp_path):
    data = DescriptorMultiheadData(x=torch.ones(2), y=torch.zeros(1), head_idx=torch.tensor(3), sample_id="s")

    loaded = DescriptorMultiheadData.load(data.save(str(tmp_path / "sample.pth")))

    assert loaded.head_idx.item() == 3
    assert loaded.energies is None and loaded.element is None


def test_multihead_batches_leave_fields_that_samples_lack_empty(tmp_path):
    dataset = create("descriptor_multihead", tmp_path, [water("a", [spectrum(), None, None], subdir_id=0)])
    sample = DescriptorMultiheadData(x=torch.ones(2), y=torch.zeros(1), head_idx=torch.tensor(0), sample_id="s")

    batch = dataset.collate_fn([sample, sample])

    assert batch.x.shape == (2, 2)
    assert batch.energies is None and batch.element is None
