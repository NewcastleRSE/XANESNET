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

"""Unit tests for the ``mlp`` and multi-head ``mh_mlp`` / ``mh_cnn`` models."""

import pytest
import torch
from torch import nn
from typing import cast

from xanesnet.models import ModelRegistry

MLP = {
    "in_size": 6,
    "out_size": 4,
    "hidden_size": 16,
    "dropout": 0.0,
    "num_hidden_layers": 3,
    "shrink_rate": 0.5,
    "activation": "relu",
}
MH_MLP = {
    "in_size": 6,
    "out_size": [4, 4, 4],
    "hidden_size": 16,
    "dropout": 0.0,
    "num_hidden_layers": 2,
    "shrink_rate": 0.5,
    "activation": "relu",
    "head_num_hidden_layers": 1,
    "head_hidden_size": 8,
    "head_shrink_rate": 1.0,
}
MH_CNN = {
    "in_size": 16,
    "out_size": [4, 4, 4],
    "dropout": 0.0,
    "num_conv_layers": 2,
    "activation": "relu",
    "out_channel": 2,
    "channel_mul": 2,
    "kernel_size": 3,
    "stride": 1,
    "head_num_hidden_layers": 1,
    "head_hidden_size": 8,
    "head_shrink_rate": 1.0,
}
MODELS = {"mlp": MLP, "mh_mlp": MH_MLP, "mh_cnn": MH_CNN}


def create(key: str, **overrides):
    return ModelRegistry.create(key, model_type=key, **{**MODELS[key], **overrides})


def input_size(key: str) -> int:
    return cast(int, MODELS[key]["in_size"])


def linear_sizes(model: nn.Module) -> list[tuple[int, int]]:
    return [(m.in_features, m.out_features) for m in model.modules() if isinstance(m, nn.Linear)]


###############################################################################
################################ SHARED CONTRACT ##############################
###############################################################################


@pytest.mark.parametrize("key", MODELS)
def test_the_signature_contains_everything_needed_to_rebuild_the_model(key):
    model = create(key)

    rebuilt = ModelRegistry.create(key, **model.signature.as_kwargs())

    assert model.signature.as_dict() == {"model_type": key, **MODELS[key]}
    assert rebuilt.signature.as_dict() == model.signature.as_dict()
    assert [p.shape for p in rebuilt.parameters()] == [p.shape for p in model.parameters()]


@pytest.mark.parametrize("key", MODELS)
def test_a_rebuilt_model_accepts_the_saved_weights(key):
    model = create(key)
    rebuilt = ModelRegistry.create(key, **model.signature.as_kwargs())

    rebuilt.load_state_dict(model.state_dict())

    model.eval()
    rebuilt.eval()
    x = torch.randn(5, input_size(key))
    torch.testing.assert_close(rebuilt(x), model(x))


@pytest.mark.parametrize("key", MODELS)
def test_weights_and_biases_follow_the_selected_initialisers(key):
    model = create(key)

    model.init_weights("normal", "ones", mean=0.0, std=0.5)

    layers = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv1d))]
    assert layers
    for layer in layers:
        assert layer.bias is not None
        assert torch.equal(layer.bias, torch.ones_like(layer.bias))
        assert layer.weight.std().item() > 0


@pytest.mark.parametrize("key", MODELS)
def test_unknown_initialisers_are_reported(key):
    with pytest.raises(KeyError):
        create(key).init_weights("not_an_initialiser", "zeros")


###############################################################################
##################################### MLP #####################################
###############################################################################


def test_hidden_layers_shrink_geometrically():
    model = create("mlp")

    assert linear_sizes(model) == [(6, 16), (16, 8), (8, 4), (4, 4)]


def test_predictions_have_one_value_per_output_for_every_sample():
    assert create("mlp")(torch.randn(7, 6)).shape == (7, 4)


def test_a_shrink_rate_that_collapses_a_hidden_layer_is_rejected():
    with pytest.raises(ValueError, match="Hidden layer 2 size is less than 1"):
        create("mlp", hidden_size=2, shrink_rate=0.3, num_hidden_layers=3)


def test_an_mlp_without_hidden_layers_is_a_single_linear_map():
    assert linear_sizes(create("mlp", num_hidden_layers=0)) == [(6, 4)]


def test_unknown_activations_are_reported():
    with pytest.raises(KeyError):
        create("mlp", activation="not_an_activation")


###############################################################################
################################ MULTI-HEAD MODELS ############################
###############################################################################


@pytest.mark.parametrize("key", ["mh_mlp", "mh_cnn"])
def test_all_heads_are_evaluated_by_default(key):
    model = create(key).eval()

    out = model(torch.randn(5, input_size(key)))

    assert out.shape == (3, 5, 4)  # heads x batch x outputs


@pytest.mark.parametrize("key", ["mh_mlp", "mh_cnn"])
def test_a_single_head_can_be_selected_and_matches_its_slice_of_the_full_output(key):
    model = create(key).eval()
    x = torch.randn(5, input_size(key))

    everything = model(x)

    for head in range(3):
        torch.testing.assert_close(model(x, active_head_idx=head), everything[head])


@pytest.mark.parametrize("key", ["mh_mlp", "mh_cnn"])
def test_heads_have_independent_parameters(key):
    model = create(key)
    first, second = model.heads[0], model.heads[1]

    assert all(a is not b for a, b in zip(first.parameters(), second.parameters()))


def test_mh_mlp_shares_the_trunk_and_gives_every_head_its_own_output_size():
    model = create("mh_mlp", out_size=[2, 5])

    assert linear_sizes(model.dense_layers) == [(6, 16), (16, 8)]
    assert model(torch.randn(3, 6), active_head_idx=0).shape == (3, 2)
    assert model(torch.randn(3, 6), active_head_idx=1).shape == (3, 5)


def test_mh_mlp_rejects_a_collapsing_trunk():
    with pytest.raises(ValueError, match="Hidden layer 2 size is less than 1"):
        create("mh_mlp", hidden_size=2, shrink_rate=0.3, num_hidden_layers=3)


def test_mh_cnn_sizes_its_heads_from_the_convolution_output():
    model = create("mh_cnn")

    # two valid convolutions with kernel 3 shrink 16 samples to 12, with 2 * 2 = 4 channels
    expected_features = 4 * 12
    assert linear_sizes(model.heads[0])[0][0] == expected_features


def test_mh_cnn_restores_the_training_mode_after_probing_its_output_size():
    model = create("mh_cnn")
    assert model.conv_layers.training is True
