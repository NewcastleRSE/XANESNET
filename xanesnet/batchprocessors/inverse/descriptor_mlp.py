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

"""Batch processor for the descriptor + MLP model combination (inverse prediction)."""

import numpy as np
import torch

from xanesnet.datasets import DescriptorData

from ..registry import BatchProcessorRegistry
from .base import InverseBatchProcessor


@BatchProcessorRegistry.register(("descriptor_inverse", "mlp"))
@BatchProcessorRegistry.register(("descriptor_inverse_mp", "mlp"))
class InverseDescriptorMLPBatchProcessor(InverseBatchProcessor):
    """Batch processor for ``DescriptorData`` + MLP (inverse prediction).

    Inverse: spectra -> descriptors. The spectral input (stored under ``x``)
    is encoded by
    :class:`~xanesnet.batchprocessors.inverse.base.InverseBatchProcessor` via
    :meth:`~xanesnet.batchprocessors.inverse.base.InverseBatchProcessor.encode_input`;
    targets and predictions are structural descriptors and pass through
    unchanged.
    """

    def input_preparation(self, batch: DescriptorData) -> dict[str, torch.Tensor]:
        """Prepare raw spectral inputs for the MLP model.

        Args:
            batch: Collated descriptor batch where ``x`` holds spectral
                intensities in inverse mode.

        Returns:
            Dict with ``"x"`` containing the raw spectral tensor.
            ``(batch_size, n_energies)``.
        """
        return {"x": batch.x}  # type: ignore[dict-item]

    def target_preparation(self, batch: DescriptorData) -> torch.Tensor:
        """Prepare descriptor targets from a descriptor batch.

        Args:
            batch: Collated descriptor batch (``y`` holds descriptor features
                in inverse mode).

        Returns:
            Descriptor feature tensor. ``(batch_size, n_descriptor_features)``.
        """
        return batch.y  # type: ignore[return-value]

    def element_preparation(self, batch: DescriptorData) -> torch.Tensor | None:
        """Prepare target-site atomic numbers from a descriptor batch.

        Args:
            batch: Collated descriptor batch.

        Returns:
            Per-sample target-site atomic numbers ``(batch_size,)``, or
            ``None`` if the dataset was built without element information.
        """
        return batch.element

    def sample_id_preparation(self, batch: DescriptorData) -> np.ndarray:
        """Prepare file names from a descriptor batch.

        Args:
            batch: Collated descriptor batch.

        Returns:
            Array of file name strings. ``(batch_size,)``.
        """
        return np.array(batch.sample_id, dtype=str)
