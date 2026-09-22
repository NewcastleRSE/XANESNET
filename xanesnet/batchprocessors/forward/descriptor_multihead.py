# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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
#   ...

"""Forward batch processor for descriptor-based multi-head models."""

from typing import cast

import numpy as np
import torch

from xanesnet.datasets import DescriptorMultiheadData

from ..registry import BatchProcessorRegistry
from .base import ForwardBatchProcessor


@BatchProcessorRegistry.register(("descriptor_multihead", "mh_mlp"))
@BatchProcessorRegistry.register(("descriptor_multihead_mp", "mh_mlp"))
@BatchProcessorRegistry.register(("descriptor_multihead", "mh_cnn"))
@BatchProcessorRegistry.register(("descriptor_multihead_mp", "mh_cnn"))
class DescriptorMultiheadBatchProcessor(ForwardBatchProcessor):
    """Batch processor for ``DescriptorMultiheadData`` + multi-head models.

    Forward: descriptors -> spectra. Descriptor inputs are forwarded as-is;
    spectral targets and predictions are encoded/decoded by
    :class:`~xanesnet.batchprocessors.forward.base.ForwardBatchProcessor`.
    Multi-head predictions are selected using each sample's ``head_idx``.
    """

    def input_preparation(self, batch: DescriptorMultiheadData) -> dict[str, torch.Tensor]:
        """Prepare model inputs from a descriptor multi-head batch.

        Args:
            batch: Collated descriptor multi-head batch.

        Returns:
            Dictionary with ``"x"`` containing the descriptor tensor. ``(batch_size, n_features)``.
        """
        return {"x": batch.x}  # type: ignore[dict-item]

    def target_preparation(self, batch: DescriptorMultiheadData) -> torch.Tensor:
        """Prepare raw spectral targets from a descriptor multi-head batch.

        Args:
            batch: Collated descriptor multi-head batch.

        Returns:
            Raw spectral intensity tensor. ``(batch_size, n_energies)``.
        """
        return batch.y  # type: ignore[return-value]

    def element_preparation(self, batch: DescriptorMultiheadData) -> torch.Tensor | None:
        """Extract target-site atomic numbers from a descriptor multi-head batch.

        Args:
            batch: Collated descriptor multi-head batch.

        Returns:
            Per-sample target-site atomic numbers ``(batch_size,)``, or
            ``None`` if the dataset was built without element information.
        """
        return batch.element

    def sample_id_extraction(self, batch: DescriptorMultiheadData) -> np.ndarray:
        """Extract file names from a descriptor multi-head batch.

        Args:
            batch: Collated descriptor multi-head batch.

        Returns:
            Array of file name strings. ``(batch_size,)``.
        """
        return np.array(batch.sample_id, dtype=str)

    def prediction_preparation(self, batch: DescriptorMultiheadData, predictions: torch.Tensor) -> torch.Tensor:
        """Select each sample's active head from stacked predictions.

        Args:
            batch: Collated multi-head batch carrying ``head_idx``.
            predictions: Stacked model predictions with shape
                ``(num_heads, batch_size, n_features)``.

        Returns:
            Predictions selected for each sample. ``(batch_size, n_features)``.
        """
        head_idx = cast(torch.Tensor, batch.head_idx)
        predictions = predictions.permute(1, 0, 2)
        row_idx = torch.arange(predictions.shape[0], device=predictions.device)
        return predictions[row_idx, head_idx]
