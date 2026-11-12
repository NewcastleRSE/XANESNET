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

"""Unit tests for the ``identity``/``none``, ``fourier`` and ``gaussian`` encodings."""

import pytest
import torch

from xanesnet.encodings import SpectraEncodingRegistry
from xanesnet.utils.exceptions import ConfigError

X = torch.sin(torch.linspace(0, 3, 24)).repeat(2, 1) + 1.5


###############################################################################
################################### IDENTITY ##################################
###############################################################################


@pytest.mark.parametrize("key", ["identity", "none"])
def test_identity_encodings_pass_spectra_through_unchanged(key):
    encoding = SpectraEncodingRegistry.create(key, encoding_type=key)

    assert encoding.encode(X) is X
    assert encoding.decode(X) is X
    assert encoding.output_size(24) == 24
    assert encoding.signature[0].as_dict() == {"encoding_type": key}


###############################################################################
#################################### FOURIER ##################################
###############################################################################


@pytest.mark.parametrize(("concat", "factor"), [(False, 2), (True, 3)])
def test_fourier_width_includes_the_original_signal_only_when_concatenating(concat, factor):
    encoding = SpectraEncodingRegistry.create("fourier", encoding_type="fourier", concat=concat)

    assert encoding.output_size(24) == factor * 24
    assert encoding.encode(X).shape == (2, factor * 24)


@pytest.mark.parametrize("concat", [False, True])
def test_fourier_decode_inverts_encode(concat):
    encoding = SpectraEncodingRegistry.create("fourier", encoding_type="fourier", concat=concat)
    torch.testing.assert_close(encoding.decode(encoding.encode(X)), X, atol=1e-5, rtol=1e-5)


def test_fourier_signature_rebuilds_the_encoding():
    encoding = SpectraEncodingRegistry.create("fourier", encoding_type="fourier", concat=True)
    (signature,) = encoding.signature

    assert signature.as_dict() == {"encoding_type": "fourier", "concat": True}
    assert SpectraEncodingRegistry.create("fourier", **signature.as_dict()).output_size(24) == 72


###############################################################################
#################################### GAUSSIAN #################################
###############################################################################


def make_gaussian(**overrides):
    kwargs = {"widths": [1.0, 2.0], "basis_stride": 2, "num_points": 24, "nonneg_output": False}
    kwargs.update(overrides)
    return SpectraEncodingRegistry.create("gaussian", encoding_type="gaussian", **kwargs)


def test_gaussian_width_is_one_coefficient_per_width_and_strided_position():
    encoding = make_gaussian()

    assert encoding.output_size(24) == 2 * 12
    assert encoding.encode(X).shape == (2, 24)
    assert encoding.decode(encoding.encode(X)).shape == X.shape


def test_gaussian_stride_rounds_up_for_widths_that_are_not_multiples():
    assert make_gaussian(basis_stride=5).output_size(24) == 2 * 5


def test_gaussian_reconstructs_a_smooth_spectrum():
    encoding = make_gaussian(widths=[1.0, 2.0, 4.0], basis_stride=1)

    reconstructed = encoding.decode(encoding.encode(X))

    assert (reconstructed - X).abs().max() < 0.1 * X.abs().max()


def test_gaussian_can_clamp_negative_output():
    encoding = make_gaussian(nonneg_output=True)
    assert encoding.decode(-torch.ones(1, 24)).min() >= 0

    unclamped = make_gaussian(nonneg_output=False)
    assert unclamped.decode(-torch.ones(1, 24)).min() < 0


def test_gaussian_rejects_a_spectrum_of_the_wrong_width():
    with pytest.raises(ValueError, match="expected input width 24"):
        make_gaussian().output_size(10)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"widths": []}, "at least one basis width"),
        ({"basis_stride": 0}, "basis_stride must be positive"),
        ({"num_points": 0}, "num_points must be positive"),
    ],
)
def test_gaussian_validates_its_configuration(overrides, message):
    with pytest.raises(ConfigError, match=message):
        make_gaussian(**overrides)


def test_gaussian_signature_rebuilds_the_encoding():
    encoding = make_gaussian(nonneg_output=True)
    (signature,) = encoding.signature

    assert signature.as_dict() == {
        "encoding_type": "gaussian",
        "widths": [1.0, 2.0],
        "basis_stride": 2,
        "num_points": 24,
        "nonneg_output": True,
    }
    rebuilt = SpectraEncodingRegistry.create("gaussian", **signature.as_dict())
    torch.testing.assert_close(rebuilt.encode(X), encoding.encode(X))
