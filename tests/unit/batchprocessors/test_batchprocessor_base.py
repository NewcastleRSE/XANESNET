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

"""Unit tests for shared forward and inverse batch-processor behavior."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from xanesnet.batchprocessors import (
    BatchProcessor,
    ForwardBatchProcessor,
    InverseBatchProcessor,
)


class RecordingEncoding:
    """Encoding stub that scales tensors and records element metadata."""

    def __init__(self) -> None:
        self.elements: list[torch.Tensor | None] = []

    def encode(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        self.elements.append(elements)
        return targets * 2

    def decode(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        self.elements.append(elements)
        return predictions / 2


class MinimalForward(ForwardBatchProcessor):
    """Forward processor stub that extracts toy inputs and targets from a batch."""

    def input_preparation(self, batch: Any) -> dict[str, Any]:
        return {"x": batch.x}

    def target_preparation(self, batch: Any) -> torch.Tensor:
        return batch.y

    def sample_id_preparation(self, batch: Any) -> np.ndarray:
        return np.array(batch.sample_id)


class MinimalInverse(InverseBatchProcessor):
    """Inverse processor stub reusing the forward test batch fields."""

    input_preparation = MinimalForward.input_preparation
    target_preparation = MinimalForward.target_preparation
    sample_id_preparation = MinimalForward.sample_id_preparation


SPECTRA = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
ELEMENTS = torch.tensor([26, 29])


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
        """Single-sample dataset that collates values into a namespace."""

        def __getitem__(self, index: int) -> dict[str, int]:
            return {"index": index}

        def collate_fn(self, samples: list[dict[str, int]]) -> SimpleNamespace:
            (sample,) = samples
            return SimpleNamespace(
                x=sample["index"] * 10,
                y=torch.tensor(sample["index"] * 100),
                sample_id=[str(sample["index"])],
            )

    processor = MinimalForward()

    assert processor.input_preparation_single(Dataset(), 3) == {"x": 30}  # type: ignore[arg-type]
    torch.testing.assert_close(processor.target_preparation_single(Dataset(), 3), torch.tensor(300))  # type: ignore[arg-type]
    assert processor.element_preparation_single(Dataset(), 3) is None  # type: ignore[arg-type]


def test_forward_processors_encode_targets_and_decode_predictions_with_elements():
    encoding = RecordingEncoding()
    processor = MinimalForward(encoding)  # type: ignore[arg-type]

    torch.testing.assert_close(processor.encode_target(SPECTRA, ELEMENTS), SPECTRA * 2)
    torch.testing.assert_close(processor.decode_target(SPECTRA, ELEMENTS), SPECTRA / 2)
    assert all(elements is ELEMENTS for elements in encoding.elements)


def test_forward_processors_leave_inputs_alone():
    inputs = {"x": SPECTRA}
    assert MinimalForward(RecordingEncoding()).encode_input(inputs, ELEMENTS) is inputs  # type: ignore[arg-type]


def test_forward_processors_without_encoding_are_the_identity():
    processor = MinimalForward()

    assert processor.encode_target(SPECTRA) is SPECTRA
    assert processor.decode_target(SPECTRA) is SPECTRA


def test_inverse_processors_encode_spectral_inputs_without_modifying_original_inputs():
    encoding = RecordingEncoding()
    inputs = {"x": SPECTRA, "other": ELEMENTS}

    encoded = MinimalInverse(encoding).encode_input(inputs, ELEMENTS)  # type: ignore[arg-type]

    torch.testing.assert_close(encoded["x"], SPECTRA * 2)
    assert encoded["other"] is ELEMENTS
    assert inputs["x"] is SPECTRA
    assert encoding.elements == [ELEMENTS]


def test_inverse_processors_use_the_configured_spectra_input_key():
    class Custom(MinimalInverse):
        """Inverse processor whose encoded spectrum uses a custom field name."""

        spectra_input_key = "spectrum"

    encoded = Custom(RecordingEncoding()).encode_input({"spectrum": SPECTRA, "x": SPECTRA})  # type: ignore[arg-type]

    torch.testing.assert_close(encoded["spectrum"], SPECTRA * 2)
    assert encoded["x"] is SPECTRA


def test_inverse_processors_never_transform_targets_or_predictions():
    processor = MinimalInverse(RecordingEncoding())  # type: ignore[arg-type]

    assert processor.encode_target(SPECTRA) is SPECTRA
    assert processor.decode_target(SPECTRA) is SPECTRA


def test_inverse_processors_without_encoding_are_the_identity():
    inputs = {"x": SPECTRA}
    assert MinimalInverse().encode_input(inputs) is inputs
