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

"""Unit tests for the batch-processor registry and the shared forward/inverse base behaviour."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xanesnet.batchprocessors import (
    BatchProcessor,
    BatchProcessorRegistry,
    ForwardBatchProcessor,
    InverseBatchProcessor,
)


class RecordingEncoding:
    """Encoding that doubles on encode and halves on decode, recording the elements it receives."""

    def __init__(self) -> None:
        self.elements: list[torch.Tensor | None] = []

    def encode(self, spectra: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        self.elements.append(elements)
        return spectra * 2

    def decode(self, spectra: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        self.elements.append(elements)
        return spectra / 2


class MinimalForward(ForwardBatchProcessor):
    def input_preparation(self, batch):
        return {"x": batch.x}

    def target_preparation(self, batch):
        return batch.y

    def sample_id_preparation(self, batch):
        return np.array(batch.sample_id)


class MinimalInverse(InverseBatchProcessor):
    input_preparation = MinimalForward.input_preparation
    target_preparation = MinimalForward.target_preparation
    sample_id_preparation = MinimalForward.sample_id_preparation


SPECTRA = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
ELEMENTS = torch.tensor([26, 29])


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


###############################################################################
################################ SHARED DEFAULTS ##############################
###############################################################################


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        BatchProcessor()  # type: ignore[abstract]


def test_defaults_pass_predictions_through_and_provide_no_elements():
    processor = MinimalForward()
    batch = SimpleNamespace(x=1, y=2, sample_id=["a"])

    assert processor.prediction_preparation(batch, SPECTRA) is SPECTRA
    assert processor.element_preparation(batch) is None
    assert processor.target_site_index_preparation(batch) is None
    assert processor.target_site_index_preparation(SimpleNamespace(target_site_index=5)) == 5


def test_single_sample_helpers_collate_the_sample_into_a_batch_first():
    class Dataset:
        def __getitem__(self, index):
            return {"index": index}

        def collate_fn(self, samples):
            (sample,) = samples
            return SimpleNamespace(x=sample["index"] * 10, y=sample["index"] * 100, sample_id=[str(sample["index"])])

    processor = MinimalForward()

    assert processor.input_preparation_single(Dataset(), 3) == {"x": 30}
    assert processor.target_preparation_single(Dataset(), 3) == 300
    assert processor.element_preparation_single(Dataset(), 3) is None


###############################################################################
############################### FORWARD / INVERSE #############################
###############################################################################


def test_forward_processors_encode_the_target_and_decode_predictions_with_the_elements():
    encoding = RecordingEncoding()
    processor = MinimalForward(encoding)

    torch.testing.assert_close(processor.encode_target(SPECTRA, ELEMENTS), SPECTRA * 2)
    torch.testing.assert_close(processor.decode_target(SPECTRA, ELEMENTS), SPECTRA / 2)
    assert all(elements is ELEMENTS for elements in encoding.elements)


def test_forward_processors_leave_the_inputs_alone():
    inputs = {"x": SPECTRA}
    assert MinimalForward(RecordingEncoding()).encode_input(inputs, ELEMENTS) is inputs


def test_forward_processors_without_an_encoding_are_the_identity():
    processor = MinimalForward()

    assert processor.encode_target(SPECTRA) is SPECTRA
    assert processor.decode_target(SPECTRA) is SPECTRA


def test_inverse_processors_encode_the_spectral_input_without_modifying_the_original_inputs():
    encoding = RecordingEncoding()
    inputs = {"x": SPECTRA, "other": ELEMENTS}

    encoded = MinimalInverse(encoding).encode_input(inputs, ELEMENTS)

    torch.testing.assert_close(encoded["x"], SPECTRA * 2)
    assert encoded["other"] is ELEMENTS
    assert inputs["x"] is SPECTRA
    assert encoding.elements == [ELEMENTS]


def test_inverse_processors_use_the_configured_spectra_input_key():
    class Custom(MinimalInverse):
        spectra_input_key = "spectrum"

    encoded = Custom(RecordingEncoding()).encode_input({"spectrum": SPECTRA, "x": SPECTRA})

    torch.testing.assert_close(encoded["spectrum"], SPECTRA * 2)
    assert encoded["x"] is SPECTRA


def test_inverse_processors_never_transform_targets_and_predictions():
    processor = MinimalInverse(RecordingEncoding())

    assert processor.encode_target(SPECTRA) is SPECTRA
    assert processor.decode_target(SPECTRA) is SPECTRA


def test_inverse_processors_without_an_encoding_are_the_identity():
    inputs = {"x": SPECTRA}
    assert MinimalInverse().encode_input(inputs) is inputs
