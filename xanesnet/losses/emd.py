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

"""Earth Mover's Distance (Wasserstein) loss for XANESNET."""

import torch

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("emd")
class EMDLoss(Loss):
    """Earth Mover's (Wasserstein) distance loss.

    Computes the cumulative-difference form of the discrete 1-D transport
    cost on a common, ordered, uniformly spaced spectral grid. With unit grid
    spacing this is the L1 distance between the cumulative spectra of
    ``preds`` and ``targets``. It is the balanced 1-Wasserstein distance
    between non-negative spectra only when each prediction and target pair
    has equal total mass. The implementation does not normalize spectra,
    enforce non-negativity, or account for physical energy spacing; inputs
    are therefore expected to satisfy those assumptions when a Wasserstein
    interpretation is intended.

    Args:
        loss_type: Identifier string for this loss type.
    """

    def __init__(
        self,
        loss_type: str,
    ) -> None:
        """Initialize ``EMDLoss``."""
        super().__init__(loss_type)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Compute the Earth Mover's Distance loss.

        Args:
            preds: Model output spectra ``(B, N)``. For a balanced
                Wasserstein interpretation, values must be non-negative and
                each row must have the same total mass as the corresponding
                target row.
            targets: Ground-truth spectra ``(B, N)`` on the same ordered,
                uniformly spaced grid as ``preds``. The grid spacing is
                treated as one, so physical energy units are ignored.
            reduction: ``"mean"`` sums the absolute cumulative differences
                over the grid for each sample and then averages those sample
                costs over the batch. ``"none"`` returns the unsummed
                point-wise map with shape ``(B, N)``. No reduction over the
                grid is performed for ``"none"``.

        Returns:
            Loss tensor.

        Raises:
            ValueError: If ``reduction`` is neither ``"mean"`` nor ``"none"``.
        """
        cdf_delta = torch.cumsum(preds - targets, dim=-1)
        loss_map = cdf_delta.abs()
        if reduction == "mean":
            return loss_map.sum(dim=-1).mean()
        if reduction == "none":
            return loss_map
        raise ValueError(f"Unsupported reduction: {reduction}")
