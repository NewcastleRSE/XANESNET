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

"""Unit tests for the composite encodings: ``CombinedEncoding`` (chain) and ``ConcatEncoding`` (parallel)."""

import pytest
import torch

from xanesnet.encodings import CombinedEncoding, ConcatEncoding, SpectraEncodingRegistry
from xanesnet.serialization.config import Config

X = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])


def scale(factor: float) -> Config:
    return Config({"encoding_type": "scale", "factor": [factor] * 3, "per_point": True, "per_element": False, "elements": None})


def subtract(value: float) -> Config:
    return Config({"encoding_type": "subtract_average", "average": [value] * 3, "per_element": False, "elements": None})


def fourier(concat: bool = False) -> Config:
    return Config({"encoding_type": "fourier", "concat": concat})


###############################################################################
################################ COMBINED (CHAIN) #############################
###############################################################################


def test_encodings_are_applied_in_order_and_decoded_in_reverse():
    chain = CombinedEncoding.from_configs([scale(2.0), subtract(1.0)])

    encoded = chain.encode(X)

    torch.testing.assert_close(encoded, X / 2 - 1)  # scale first, then subtract (the order matters)
    torch.testing.assert_close(chain.decode(encoded), X)


def test_output_size_and_prepare_follow_the_chain():
    chain = CombinedEncoding.from_configs([scale(2.0), fourier(concat=False)])
    chain.prepare(3)

    assert chain.output_size(3) == 6
    assert chain.encode(X).shape == (2, 6)
    torch.testing.assert_close(chain.decode(chain.encode(X)), X)


def test_elements_are_passed_to_every_member():
    per_element = Config(
        {
            "encoding_type": "subtract_average",
            "average": [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
            "per_element": True,
            "elements": [26, 29],
        }
    )
    chain = CombinedEncoding.from_configs([scale(1.0), per_element])

    encoded = chain.encode(X, torch.tensor([26, 29]))

    torch.testing.assert_close(encoded, torch.stack([X[0] - 1.0, X[1] - 5.0]))


def test_signature_lists_the_member_signatures_in_order_and_rebuilds_the_chain():
    chain = CombinedEncoding.from_configs([scale(2.0), subtract(1.0)])

    signature = chain.signature
    assert [item.get_str("encoding_type") for item in signature] == ["scale", "subtract_average"]

    rebuilt = CombinedEncoding.from_configs(signature)
    torch.testing.assert_close(rebuilt.encode(X), chain.encode(X))


@pytest.mark.parametrize("build", [lambda: CombinedEncoding([]), lambda: CombinedEncoding.from_configs([])])
def test_at_least_one_member_is_required(build):
    with pytest.raises(ValueError):
        build()


def test_unknown_member_types_are_reported():
    with pytest.raises(KeyError, match="does_not_exist"):
        CombinedEncoding.from_configs([Config({"encoding_type": "does_not_exist"})])


###############################################################################
################################ CONCAT (PARALLEL) ############################
###############################################################################


def make_concat(*members: Config) -> ConcatEncoding:
    encoding = SpectraEncodingRegistry.create("concat", encoding_type="concat", encodings=list(members))
    encoding.prepare(3)
    return encoding


def test_members_see_the_same_input_and_their_outputs_are_concatenated():
    encoding = make_concat(scale(2.0), scale(4.0))

    encoded = encoding.encode(X)

    assert encoding.output_size(3) == 6
    torch.testing.assert_close(encoded, torch.cat([X / 2, X / 4], dim=-1))


def test_decoding_averages_the_reconstruction_of_each_member():
    encoding = make_concat(scale(2.0), scale(4.0))
    first, second = torch.ones(2, 3), 3 * torch.ones(2, 3)

    decoded = encoding.decode(torch.cat([first, second], dim=-1))

    torch.testing.assert_close(decoded, (first * 2 + second * 4) / 2)
    torch.testing.assert_close(encoding.decode(encoding.encode(X)), X)


def test_members_with_different_output_widths_are_split_correctly():
    encoding = make_concat(scale(2.0), fourier(concat=False))

    assert encoding.output_size(3) == 3 + 6
    torch.testing.assert_close(encoding.decode(encoding.encode(X)), X)


@pytest.mark.parametrize("method", ["encode", "decode"])
def test_prepare_must_be_called_first(method):
    encoding = SpectraEncodingRegistry.create("concat", encoding_type="concat", encodings=[scale(2.0)])
    with pytest.raises(RuntimeError, match="before prepare"):
        getattr(encoding, method)(X)


def test_input_and_prediction_widths_are_validated():
    encoding = make_concat(scale(2.0), scale(4.0))

    with pytest.raises(ValueError, match="expected input width 3"):
        encoding.encode(torch.ones(2, 5))
    with pytest.raises(ValueError, match="expected prediction width 6"):
        encoding.decode(torch.ones(2, 5))


def test_preparing_again_rebinds_the_expected_width():
    encoding = make_concat(scale(2.0))
    encoding.prepare(3)
    with pytest.raises(ValueError, match="expected input width 3"):
        encoding.encode(torch.ones(2, 4))


def test_signature_nests_the_member_signatures_and_rebuilds_the_encoding():
    encoding = make_concat(scale(2.0), subtract(1.0))

    (signature,) = encoding.signature
    assert signature.get_str("encoding_type") == "concat"
    assert [item.get_str("encoding_type") for item in signature.get_config_list("encodings")] == [
        "scale",
        "subtract_average",
    ]

    rebuilt = SpectraEncodingRegistry.create("concat", **signature.as_kwargs())
    rebuilt.prepare(3)
    torch.testing.assert_close(rebuilt.encode(X), encoding.encode(X))


def test_at_least_one_member_is_required_for_concat():
    with pytest.raises(ValueError, match="at least one"):
        SpectraEncodingRegistry.create("concat", encoding_type="concat", encodings=[])
