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

"""Unit tests for the ``"auto"`` resolvers: encoding parameters derived from training-set spectra statistics."""

import pytest
import torch

from xanesnet.serialization.auto_config.registries import EncodingAutoResolver
from xanesnet.serialization.auto_config.statistics import SpectralStatisticsCollector

# element 26: rows [0, 2] and [2, 4]; element 29: two identical rows [10, 20]
SPECTRA = torch.tensor([[0.0, 2.0], [2.0, 4.0], [10.0, 20.0], [10.0, 20.0]])
ELEMENTS = torch.tensor([26, 26, 29, 29])


@pytest.fixture(scope="module")
def statistics() -> SpectralStatisticsCollector:
    collector = SpectralStatisticsCollector()
    collector.update(SPECTRA, ELEMENTS)
    return collector


def resolve(encoding_type: str, statistics, **item):
    return EncodingAutoResolver.get(encoding_type)({"encoding_type": encoding_type, **item}, statistics)


###############################################################################
################################### OVERALL ###################################
###############################################################################


def test_overall_statistics_are_used_per_point_by_default(statistics):
    assert resolve("z_score", statistics) == {
        "mean": pytest.approx([5.5, 11.5]),
        "std": pytest.approx(SPECTRA.double().std(0, unbiased=False).tolist()),
    }
    assert resolve("min_max", statistics) == {"minimum": [0.0, 2.0], "maximum": [10.0, 20.0]}
    assert resolve("subtract_average", statistics) == {"average": pytest.approx([5.5, 11.5])}
    assert resolve("scale", statistics) == {"factor": pytest.approx(SPECTRA.double().std(0, unbiased=False).tolist())}


def test_per_point_false_collapses_each_parameter_to_a_single_global_value(statistics):
    resolved = resolve("min_max", statistics, per_point=False)
    assert resolved == {"minimum": [0.0], "maximum": [20.0]}

    resolved = resolve("z_score", statistics, per_point=False)
    assert resolved["mean"] == pytest.approx([float(SPECTRA.mean())])
    assert len(resolved["std"]) == 1


###############################################################################
################################## PER ELEMENT ################################
###############################################################################


def test_per_element_resolves_one_row_per_known_element_in_sorted_order(statistics):
    resolved = resolve("z_score", statistics, per_element=True)

    assert resolved["elements"] == [26, 29]
    assert resolved["mean"] == [[1.0, 3.0], [10.0, 20.0]]
    assert resolved["std"] == [[1.0, 1.0], [0.0, 0.0]]


def test_per_element_honours_an_explicit_element_list_and_its_order(statistics):
    resolved = resolve("min_max", statistics, per_element=True, elements=[29, 26])

    assert resolved["elements"] == [29, 26]
    assert resolved["minimum"] == [[10.0, 20.0], [0.0, 2.0]]
    assert resolved["maximum"] == [[10.0, 20.0], [2.0, 4.0]]


def test_per_element_with_per_point_false_gives_single_value_rows(statistics):
    resolved = resolve("min_max", statistics, per_element=True, per_point=False)

    assert resolved["minimum"] == [[0.0], [10.0]]
    assert resolved["maximum"] == [[4.0], [20.0]]


def test_per_element_subtract_average_and_scale(statistics):
    assert resolve("subtract_average", statistics, per_element=True)["average"] == [[1.0, 3.0], [10.0, 20.0]]
    assert resolve("scale", statistics, per_element=True)["factor"] == [[1.0, 1.0], [0.0, 0.0]]


def test_unknown_requested_element_is_reported(statistics):
    with pytest.raises(Exception, match="Element 8 not found"):
        resolve("z_score", statistics, per_element=True, elements=[8])


###############################################################################
################################### GAUSSIAN ##################################
###############################################################################


def test_gaussian_resolves_the_spectrum_width(statistics):
    assert resolve("gaussian", statistics) == {"num_points": 2}
