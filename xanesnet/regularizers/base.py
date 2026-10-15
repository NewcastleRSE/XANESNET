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

"""Abstract base class for all XANESNET regularizers."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from xanesnet.models import Model


class Regularizer(nn.Module, ABC):
    """Abstract base class for all XANESNET regularizers.

    Args:
        regularizer_type: Identifier string for the concrete regularizer type.
        weight: Scalar multiplier applied to the regularization term.
    """

    def __init__(
        self,
        regularizer_type: str,
        weight: float,
    ) -> None:
        """Initialize ``Regularizer``."""
        super().__init__()

        self.regularizer_type = regularizer_type
        self.weight = weight

    @abstractmethod
    def forward(self, model: Model) -> torch.Tensor:
        """Compute the regularization penalty for a model.

        Args:
            model: The model whose parameters are penalised.

        Returns:
            Scalar regularization loss tensor.
        """
        ...
