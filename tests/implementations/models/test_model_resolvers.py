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

"""Unit tests for the model-specific auto-config resolvers (``"auto"`` model fields derived from the data)."""

from types import SimpleNamespace

import pytest
import torch

from xanesnet.models import ModelRegistry
from xanesnet.serialization.auto_config.registries import ModelAutoResolver

TARGET = torch.zeros(5, 7)
FEATURES = torch.zeros(5, 11)


def basis(groups: int, coefficients: int) -> SimpleNamespace:
    return SimpleNamespace(widths_eV=[1.0] * groups, Phi=torch.zeros(3, coefficients))


@pytest.mark.parametrize(
    ("key", "inputs", "dataset", "expected"),
    [
        ("mlp", {"x": FEATURES}, None, {"in_size": 11, "out_size": 7}),
        ("mh_mlp", {"x": FEATURES}, SimpleNamespace(num_heads=3), {"in_size": 11, "out_size": [7, 7, 7]}),
        ("mh_cnn", {"x": FEATURES}, SimpleNamespace(num_heads=2), {"in_size": 11, "out_size": [7, 7]}),
        ("schnet", {}, None, {"reduce_channels_2": 7}),
        ("dimenet", {}, None, {"out_channels": 7}),
        ("dimenet++", {}, None, {"out_channels": 7}),
        ("e3ee", {}, None, {"out_size": 7}),
        ("e3ee_full", {}, None, {"out_size": 7}),
        ("gemnet", {}, None, {"num_targets": 7}),
        ("gemnet_oc", {}, None, {"num_targets": 7}),
        (
            "envembed",
            {"descriptor_features": FEATURES, "basis": basis(groups=3, coefficients=12)},
            None,
            {"in_size": 11, "kgroups": [4, 4, 4]},
        ),
    ],
)
def test_resolvers_derive_model_sizes_from_the_prepared_batch(key, inputs, dataset, expected):
    assert ModelAutoResolver.get(key)(inputs, TARGET, dataset) == expected


def test_every_shipped_model_has_a_resolver():
    shipped = {key for key in ModelRegistry.list() if not key.startswith("unit_test_")}  # skip test doubles

    assert shipped
    assert shipped <= set(ModelAutoResolver.list())
