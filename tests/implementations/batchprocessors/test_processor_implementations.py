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

"""Unit tests for the model-specific batch processors: which batch fields feed the model and how outputs are mapped back."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xanesnet.batchprocessors import BatchProcessorRegistry

X = torch.tensor([[1, 2], [3, 4], [5, 6]])
INTENSITIES = torch.arange(6.0).reshape(3, 2)
TARGET_SITE_MASK = torch.tensor([True, False, True])


def processor_for(dataset_type: str, model_type: str):
    return BatchProcessorRegistry.create((dataset_type, model_type))


def batch(**fields) -> SimpleNamespace:
    return SimpleNamespace(sample_id=[1, 2, 3], **fields)


###############################################################################
################################ DESCRIPTOR MODELS ############################
###############################################################################


@pytest.mark.parametrize(
    ("dataset_type", "model_type"),
    [("descriptor", "mlp"), ("descriptor_multihead", "mh_mlp"), ("descriptor_inverse", "mlp")],
)
def test_descriptor_processors_read_features_targets_and_elements(dataset_type, model_type):
    processor = processor_for(dataset_type, model_type)
    data = batch(x=X, y=INTENSITIES, element=torch.tensor([26, 29, 26]))

    assert processor.input_preparation(data) == {"x": X}
    assert processor.target_preparation(data) is INTENSITIES
    assert processor.element_preparation(data) is data.element
    assert list(processor.sample_id_preparation(data)) == ["1", "2", "3"]
    assert processor.sample_id_preparation(data).dtype.kind == "U"


@pytest.mark.parametrize("dataset_type", ["descriptor_multihead", "descriptor_multihead_inverse"])
def test_multihead_predictions_are_reduced_to_the_head_of_each_sample(dataset_type):
    processor = processor_for(dataset_type, "mh_mlp")
    heads, samples, width = 2, 3, 4
    predictions = torch.arange(heads * samples * width, dtype=torch.float32).reshape(heads, samples, width)
    data = batch(head_idx=torch.tensor([1, 0, 1]))

    selected = processor.prediction_preparation(data, predictions)

    assert selected.shape == (samples, width)
    torch.testing.assert_close(selected[0], predictions[1, 0])
    torch.testing.assert_close(selected[1], predictions[0, 1])
    torch.testing.assert_close(selected[2], predictions[1, 2])


###############################################################################
################################# GRAPH MODELS ################################
###############################################################################

GRAPH_FIELDS = {
    ("geometrygraph", "schnet"): ["z", "edge_index", "edge_weight"],
    ("geometrygraph", "dimenet"): ["z", "edge_index", "edge_weight", "angle", "idx_kj", "idx_ji"],
    ("geometrygraph", "dimenet++"): ["z", "edge_index", "edge_weight", "angle", "idx_kj", "idx_ji"],
}


@pytest.mark.parametrize(("keys", "expected"), GRAPH_FIELDS.items())
def test_geometry_graph_models_receive_the_atoms_and_edges(keys, expected):
    processor = processor_for(*keys)
    data = batch(
        x=X,
        edge_index=1,
        edge_weight=2,
        angle=3,
        idx_kj=4,
        idx_ji=5,
        intensities=INTENSITIES,
        target_site_mask=TARGET_SITE_MASK,
    )

    inputs = processor.input_preparation(data)

    assert list(inputs) == expected
    assert inputs["z"] is X


@pytest.mark.parametrize(
    "keys",
    [("geometrygraph", "schnet"), ("geometrygraph", "dimenet"), ("gemnet", "gemnet"), ("gemnet_oc", "gemnet_oc")],
)
def test_per_atom_models_are_evaluated_at_the_target_sites(keys):
    processor = processor_for(*keys)
    data = batch(x=X, intensities=INTENSITIES, target_site_mask=TARGET_SITE_MASK)
    per_atom_predictions = INTENSITIES + 100

    torch.testing.assert_close(processor.prediction_preparation(data, per_atom_predictions), per_atom_predictions[[0, 2]])
    torch.testing.assert_close(processor.element_preparation(data), X[[0, 2]])
    assert processor.target_preparation(data) is INTENSITIES


GEMNET_REQUIRED = dict(
    edge_vec=1, edge_weight=2, id_swap=3, id3_expand_ba=4, id3_reduce_ca=5, Kidx3=6, id_c=7, id_a=8, edge_index=9
)


def test_gemnet_optional_quadruplet_fields_default_to_none():
    inputs = processor_for("gemnet", "gemnet").input_preparation(batch(x=X, **GEMNET_REQUIRED))

    assert inputs["z"] is X
    assert inputs["id_c"] == 7
    assert inputs["Kidx4"] is None
    assert inputs["int_edge_vec"] is None


def test_gemnet_optional_fields_are_forwarded_when_present():
    inputs = processor_for("gemnet", "gemnet").input_preparation(batch(x=X, Kidx4=42, **GEMNET_REQUIRED))
    assert inputs["Kidx4"] == 42


def test_gemnet_oc_optional_interaction_fields_default_to_none_and_are_forwarded_when_present():
    processor = processor_for("gemnet_oc", "gemnet_oc")

    without = processor.input_preparation(batch(x=X, **GEMNET_REQUIRED))
    with_extra = processor.input_preparation(batch(x=X, a2a_edge_index=11, **GEMNET_REQUIRED))

    assert without["a2a_edge_index"] is None
    assert without["qint_edge_index"] is None
    assert with_extra["a2a_edge_index"] == 11


###############################################################################
################################ EQUIVARIANT MODELS ###########################
###############################################################################


def test_e3ee_takes_the_element_from_the_target_site_of_each_sample():
    processor = processor_for("e3ee", "e3ee")
    x = torch.tensor([[26, 8, 8], [29, 29, 1]])  # samples x atoms
    data = batch(x=x, target_site_index=torch.tensor([0, 1]), intensities=INTENSITIES)

    assert processor.element_preparation(data).tolist() == [26, 29]
    assert processor.target_preparation(data) is INTENSITIES


def test_e3ee_passes_the_graph_and_path_tensors_to_the_model():
    fields = [
        "x", "mask", "target_site_index", "edge_src", "edge_dst", "edge_weight", "edge_vec", "att_dst", "att_dist",
        "att_vec", "energies", "path_j", "path_k", "path_r0j", "path_r0k", "path_rjk", "path_cosangle", "path_batch",
    ]  # fmt: skip
    data = batch(**{name: index for index, name in enumerate(fields)})

    inputs = processor_for("e3ee", "e3ee").input_preparation(data)

    assert inputs == {name: index for index, name in enumerate(fields)}


def test_e3ee_full_is_evaluated_at_the_target_sites_of_the_full_graph():
    processor = processor_for("e3ee_full", "e3ee_full")
    data = batch(x=X, intensities=INTENSITIES, target_site_mask=TARGET_SITE_MASK)

    torch.testing.assert_close(processor.prediction_preparation(data, INTENSITIES), INTENSITIES[[0, 2]])
    torch.testing.assert_close(processor.element_preparation(data), X[[0, 2]])
    assert processor.target_preparation(data) is INTENSITIES


def test_e3ee_full_passes_the_graph_energy_grid_and_path_tensors_to_the_model():
    fields = [
        "x", "mask", "target_site_mask", "edge_src", "edge_dst", "edge_weight", "edge_vec", "att_src", "att_dst",
        "att_dist", "att_vec", "energies", "path_center", "path_j", "path_k", "path_r0j", "path_r0k", "path_rjk",
        "path_cosangle",
    ]  # fmt: skip
    expected = {name: index for index, name in enumerate(fields)}

    assert processor_for("e3ee_full", "e3ee_full").input_preparation(batch(**expected)) == expected


def test_envembed_forwards_descriptors_and_spectral_basis():
    data = batch(
        descriptor_features="d", distance_features="r", lengths="l", basis="b", intensities=INTENSITIES, element="e"
    )
    processor = processor_for("envembed", "envembed")

    assert processor.input_preparation(data) == {
        "descriptor_features": "d",
        "distance_features": "r",
        "lengths": "l",
        "basis": "b",
    }
    assert processor.target_preparation(data) is INTENSITIES
    assert processor.element_preparation(data) == "e"
    assert isinstance(processor.sample_id_preparation(data), np.ndarray)
