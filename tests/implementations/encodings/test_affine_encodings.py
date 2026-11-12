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

"""Unit tests for the affine encodings: ``scale``, ``min_max``, ``z_score`` and ``subtract_average``."""

import pytest
import torch

from xanesnet.encodings import SpectraEncodingRegistry
from xanesnet.utils.exceptions import ConfigError

X = torch.tensor([[3.0, 5.0, 7.0], [4.0, 6.0, 8.0]])

# family -> (constructor arguments, encoded value of X[0])
FAMILIES = {
    "scale": ({"factor": [2.0, 4.0, 5.0]}, [1.5, 1.25, 1.4]),
    "min_max": ({"minimum": [0.0, 1.0, 2.0], "maximum": [2.0, 5.0, 12.0]}, [1.5, 1.0, 0.5]),
    "z_score": ({"mean": [1.0, 2.0, 3.0], "std": [2.0, 4.0, 5.0]}, [1.0, 0.75, 0.8]),
    "subtract_average": ({"average": [1.0, 2.0, 3.0]}, [2.0, 3.0, 4.0]),
}
# families that have a ``per_point`` option in their constructor
HAS_PER_POINT = {"scale", "min_max", "z_score"}


def build(family: str, per_element: bool = False, elements: list[int] | None = None, **overrides):
    params, _ = FAMILIES[family]
    kwargs = dict(params)
    if per_element:  # one parameter row per element: the second row is derived from the first
        kwargs = {name: [values, [2 * v + 1 for v in values]] for name, values in kwargs.items()}
    kwargs.update(overrides)
    kwargs.update(per_element=per_element, elements=elements)
    if family in HAS_PER_POINT:
        kwargs.setdefault("per_point", True)
    return SpectraEncodingRegistry.create(family, encoding_type=family, **kwargs)


###############################################################################
################################### BEHAVIOUR #################################
###############################################################################


@pytest.mark.parametrize("family", FAMILIES)
def test_encode_applies_the_documented_formula(family):
    expected = torch.tensor(FAMILIES[family][1])
    torch.testing.assert_close(build(family).encode(X)[0], expected)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("per_element", [False, True])
def test_decode_inverts_encode(family, per_element):
    encoding = build(family, per_element=per_element, elements=[26, 29] if per_element else None)
    elements = torch.tensor([26, 29]) if per_element else None

    torch.testing.assert_close(encoding.decode(encoding.encode(X, elements), elements), X)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_encoded_width_equals_the_input_width(family):
    assert build(family).output_size(3) == 3


@pytest.mark.parametrize("family", ["scale", "min_max", "z_score"])
def test_a_zero_spread_is_treated_as_one_instead_of_dividing_by_zero(family):
    zero_spread = {
        "scale": {"factor": [0.0, 0.0, 0.0]},
        "min_max": {"minimum": [1.0, 2.0, 3.0], "maximum": [1.0, 2.0, 3.0]},
        "z_score": {"mean": [1.0, 2.0, 3.0], "std": [0.0, 0.0, 0.0]},
    }[family]
    encoding = build(family, **zero_spread)

    encoded = encoding.encode(X)
    assert torch.isfinite(encoded).all()
    torch.testing.assert_close(encoding.decode(encoded), X)


###############################################################################
################################ PER-ELEMENT MODE #############################
###############################################################################


@pytest.mark.parametrize("family", FAMILIES)
def test_each_sample_uses_the_parameters_of_its_own_element(family):
    encoding = build(family, per_element=True, elements=[26, 29])

    encoded = encoding.encode(X, torch.tensor([29, 26]))

    for row, element_row in enumerate((1, 0)):  # sample 0 is element 29 (parameter row 1)
        single = build(family, per_element=True, elements=[26, 29])
        reference = single.encode(X[row : row + 1], torch.tensor([[26, 29][element_row]]))
        torch.testing.assert_close(encoded[row : row + 1], reference)


def test_the_element_order_of_the_configuration_defines_the_parameter_rows():
    forward = build("subtract_average", per_element=True, elements=[26, 29])
    reversed_ = build("subtract_average", per_element=True, elements=[29, 26])
    element_26 = torch.tensor([26, 26])

    assert not torch.equal(forward.encode(X, element_26), reversed_.encode(X, element_26))


def test_element_aware_encodings_require_atomic_numbers():
    encoding = build("z_score", per_element=True, elements=[26, 29])
    with pytest.raises(ConfigError, match="requires per-sample atomic numbers"):
        encoding.encode(X)


def test_elements_without_parameters_are_reported_with_their_atomic_numbers():
    encoding = build("z_score", per_element=True, elements=[26, 29])
    with pytest.raises(ConfigError, match=r"no parameters for element\(s\) \[27\]"):
        encoding.encode(X, torch.tensor([26, 27]))


@pytest.mark.parametrize("atomic_number", [100, -1])
def test_out_of_range_atomic_numbers_are_rejected(atomic_number):
    encoding = build("z_score", per_element=True, elements=[26, 29])
    with pytest.raises(ConfigError, match="out-of-range"):
        encoding.encode(X, torch.tensor([26, atomic_number]))


def test_element_independent_encodings_ignore_the_elements_argument():
    encoding = build("z_score")
    torch.testing.assert_close(encoding.encode(X, torch.tensor([26, 29])), encoding.encode(X))


###############################################################################
################################### SIGNATURE #################################
###############################################################################


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("per_element", [False, True])
def test_signature_rebuilds_an_equivalent_encoding(family, per_element):
    elements = [26, 29] if per_element else None
    original = build(family, per_element=per_element, elements=elements)

    (signature,) = original.signature
    kwargs = signature.as_dict()
    assert kwargs["encoding_type"] == family
    assert kwargs["per_element"] is per_element
    assert kwargs["elements"] == elements

    rebuilt = SpectraEncodingRegistry.create(family, **kwargs)
    element_tensor = torch.tensor([26, 29]) if per_element else None
    torch.testing.assert_close(rebuilt.encode(X, element_tensor), original.encode(X, element_tensor))


def test_signature_records_the_effective_parameters_not_the_degenerate_input():
    encoding = build("z_score", std=[0.0, 2.0, 4.0])
    assert encoding.signature[0].as_dict()["std"] == [1.0, 2.0, 4.0]


###############################################################################
################################### VALIDATION ################################
###############################################################################


@pytest.mark.parametrize(
    ("family", "overrides"),
    [
        ("scale", {"factor": []}),
        ("min_max", {"minimum": [], "maximum": []}),
        ("z_score", {"mean": [], "std": []}),
        ("subtract_average", {"average": []}),
    ],
)
def test_empty_parameters_are_rejected(family, overrides):
    with pytest.raises(ConfigError, match="non-empty"):
        build(family, **overrides)


@pytest.mark.parametrize(
    ("family", "overrides"),
    [
        ("min_max", {"minimum": [0.0, 1.0], "maximum": [1.0, 2.0, 3.0]}),
        ("z_score", {"mean": [0.0, 1.0], "std": [1.0, 2.0, 3.0]}),
    ],
)
def test_mismatched_parameter_lengths_are_rejected(family, overrides):
    with pytest.raises(ConfigError, match="mismatch"):
        build(family, **overrides)


@pytest.mark.parametrize(
    ("elements", "message"),
    [
        (None, "non-empty list of elements"),
        ([], "non-empty list of elements"),
        ([26, 26], "must be unique"),
        ([26, 29, 8], "one parameter row per element"),
    ],
)
def test_per_element_mode_validates_the_element_list(elements, message):
    with pytest.raises(ConfigError, match=message):
        build("z_score", per_element=True, elements=elements)


def test_per_element_rows_must_have_equal_length():
    with pytest.raises(ConfigError, match="equal length"):
        build("subtract_average", per_element=True, elements=[26, 29], average=[[1.0, 2.0], [1.0]])
