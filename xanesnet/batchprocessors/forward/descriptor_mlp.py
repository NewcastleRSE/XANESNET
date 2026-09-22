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

"""Batch processor for the descriptor + MLP model combination (forward prediction)."""

import numpy as np
import torch

from xanesnet.datasets import DescriptorData

from ..registry import BatchProcessorRegistry
from .base import ForwardBatchProcessor


@BatchProcessorRegistry.register(("descriptor", "mlp"))
@BatchProcessorRegistry.register(("descriptor_mp", "mlp"))
class DescriptorMLPBatchProcessor(ForwardBatchProcessor):
    """Batch processor for ``DescriptorData`` + MLP (forward prediction).

    Forward: descriptors -> spectra. Descriptor inputs are forwarded as-is;
    spectral targets and predictions are encoded/decoded by
    :class:`~xanesnet.batchprocessors.forward.base.ForwardBatchProcessor`.
    """

    def input_preparation(self, batch: DescriptorData) -> dict[str, torch.Tensor]:
        """Prepare MLP inputs from a descriptor batch.

        Args:
            batch: Collated descriptor batch.

        Returns:
            Dict with ``"x"`` containing the descriptor tensor. ``(batch_size, n_features)``.
        """
        return {"x": batch.x}  # type: ignore[dict-item]

    def target_preparation(self, batch: DescriptorData) -> torch.Tensor:
        """Prepare raw spectral targets from a descriptor batch.

        Args:
            batch: Collated descriptor batch.

        Returns:
            Raw spectral intensity tensor. ``(batch_size, n_energies)``.
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
