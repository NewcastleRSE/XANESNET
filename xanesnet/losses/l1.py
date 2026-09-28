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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""L1 (mean absolute error) loss for XANESNET."""

import torch
import torch.nn.functional as F
from torch import nn

from .base import Loss
from .registry import LossRegistry


@LossRegistry.register("l1")
class L1Loss(Loss):
    """L1 (mean absolute error) loss.

    Args:
        loss_type: Identifier string for this loss type.
    """

    def __init__(
        self,
        loss_type: str,
    ) -> None:
        """Initialize ``L1Loss``."""
        super().__init__(loss_type)

        self.loss = nn.L1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Compute the L1 (mean absolute error) loss.

        Args:
            preds: Model output predictions ``(B, N)``.
            targets: Ground-truth target values ``(B, N)``.
            reduction: ``"mean"`` returns the scalar loss; ``"none"`` returns
                the energy-resolved map with shape ``(B, N)``.

        Returns:
            Loss tensor.
        """
        return F.l1_loss(preds, targets, reduction=reduction)
