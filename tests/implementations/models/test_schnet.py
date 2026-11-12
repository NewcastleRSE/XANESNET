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

"""Unit tests for the ``schnet`` model."""

import pytest
import torch

from xanesnet.models import ModelRegistry
from xanesnet.models.schnet.schnet import GaussianSmearing, ShiftedSoftplus

CONFIG = {
    "hidden_channels": 8,
    "reduce_channels_1": 6,
    "reduce_channels_2": 4,
    "num_filters": 8,
    "num_interactions": 2,
    "num_gaussians": 10,
    "cutoff": 4.0,
    "mean_spectrum": None,
}


def create(**overrides):
    return ModelRegistry.create("schnet", model_type="schnet", **{**CONFIG, **overrides})


def graph():
    """Water: atomic numbers and a fully connected directed edge list."""
    z = torch.tensor([8, 1, 1])
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    edge_weight = torch.tensor([0.96, 0.96, 0.96, 1.5, 0.96, 1.5])
    return z, edge_index, edge_weight


def test_every_atom_gets_a_spectrum():
    out = create()(*graph())
    assert out.shape == (3, 4)


def test_the_signature_rebuilds_an_equivalent_model():
    model = create(mean_spectrum=[0.1, 0.2, 0.3, 0.4])

    rebuilt = ModelRegistry.create("schnet", **model.signature.as_kwargs())

    assert rebuilt.signature.as_dict() == {"model_type": "schnet", **CONFIG, "mean_spectrum": [0.1, 0.2, 0.3, 0.4]}
    rebuilt.load_state_dict(model.state_dict())
    torch.testing.assert_close(rebuilt(*graph()), model(*graph()))


def test_the_mean_spectrum_is_added_to_every_prediction():
    plain = create()
    shifted = create(mean_spectrum=[1.0, 2.0, 3.0, 4.0])
    shifted.load_state_dict({**plain.state_dict(), "mean_tensor": shifted.mean_tensor}, strict=True)

    difference = shifted(*graph()) - plain(*graph())

    torch.testing.assert_close(difference, torch.tensor([[1.0, 2.0, 3.0, 4.0]]).expand(3, 4))


def test_the_mean_spectrum_is_checkpointed_with_the_weights():
    assert "mean_tensor" in create(mean_spectrum=[0.0] * 4).state_dict()
    assert "mean_tensor" not in create().state_dict()


def test_the_mean_spectrum_must_match_the_output_width():
    with pytest.raises(ValueError, match="mean_spectrum length must match reduce_channels_2"):
        create(mean_spectrum=[0.0, 1.0])


def test_predictions_do_not_depend_on_the_order_of_the_atoms():
    model = create().eval()
    z, edge_index, edge_weight = graph()
    permutation = torch.tensor([2, 0, 1])  # new position -> old atom
    inverse = torch.argsort(permutation)

    out = model(z, edge_index, edge_weight)
    permuted = model(z[permutation], inverse[edge_index], edge_weight)

    torch.testing.assert_close(permuted, out[permutation], atol=1e-5, rtol=1e-5)


def test_edges_beyond_the_cutoff_do_not_contribute():
    model = create(cutoff=2.0).eval()
    z = torch.tensor([8, 1])
    near = model(z, torch.tensor([[0, 1], [1, 0]]), torch.tensor([1.0, 1.0]))

    # an extra pair at (or beyond) the cutoff gets a zero filter and does not change the result
    far_z = torch.tensor([8, 1, 1])
    far_edges = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    far = model(far_z, far_edges, torch.tensor([1.0, 1.0, 2.0, 2.0]))

    torch.testing.assert_close(far[:2], near, atol=1e-5, rtol=1e-5)


def test_init_weights_covers_embeddings_interactions_and_output_layers():
    model = create()

    model.init_weights("normal", "zeros", mean=0.0, std=0.5)

    assert torch.equal(model.lin1.bias, torch.zeros_like(model.lin1.bias))
    assert torch.equal(model.lin2.bias, torch.zeros_like(model.lin2.bias))
    assert torch.equal(model.interactions[0].lin.bias, torch.zeros_like(model.interactions[0].lin.bias))
    assert model.lin1.weight.std().item() > 0


def test_gaussian_smearing_expands_a_distance_into_a_peak_around_its_value():
    smearing = GaussianSmearing(0.0, 4.0, 5)

    values = smearing(torch.tensor([1.0, 2.5]))

    assert values.shape == (2, 5)
    assert values[0].argmax().item() == 1  # centres are 0, 1, 2, 3, 4
    assert values[0, 1].item() == pytest.approx(1.0)


def test_gaussian_smearing_needs_at_least_two_centres():
    with pytest.raises(ValueError, match="num_gaussians must be at least 2"):
        GaussianSmearing(0.0, 4.0, 1)


def test_shifted_softplus_is_zero_at_zero():
    assert ShiftedSoftplus()(torch.zeros(1)).item() == pytest.approx(0.0, abs=1e-7)
