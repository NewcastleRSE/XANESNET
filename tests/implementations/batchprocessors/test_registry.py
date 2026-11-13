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

"""Tests for the registered dataset/model batch-processor mappings."""

import pytest

from xanesnet.batchprocessors import (
    BatchProcessorRegistry,
    ForwardBatchProcessor,
    InverseBatchProcessor,
)


###############################################################################
################################### REGISTRY ##################################
###############################################################################


def test_processors_are_looked_up_by_dataset_and_model_type_ignoring_case():
    assert BatchProcessorRegistry.get(("Descriptor", "MLP")) is BatchProcessorRegistry.get(("descriptor", "mlp"))


def test_unknown_combinations_name_both_types():
    with pytest.raises(KeyError, match="for descriptor, not_a_model"):
        BatchProcessorRegistry.get(("descriptor", "not_a_model"))


def test_message_passing_datasets_share_the_processor_of_their_plain_counterpart():
    mp_keys = [key for key in BatchProcessorRegistry.list() if key[0].endswith("_mp")]

    assert mp_keys
    for dataset_type, model_type in mp_keys:
        plain = (dataset_type.removesuffix("_mp"), model_type)
        assert BatchProcessorRegistry.get((dataset_type, model_type)) is BatchProcessorRegistry.get(plain)


def test_inverse_datasets_are_handled_by_inverse_processors_only():
    for dataset_type, model_type in BatchProcessorRegistry.list():
        processor = BatchProcessorRegistry.get((dataset_type, model_type))
        assert issubclass(processor, InverseBatchProcessor) == ("inverse" in dataset_type)
        assert issubclass(processor, ForwardBatchProcessor) == ("inverse" not in dataset_type)
