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

"""L1 regularization for XANESNET."""

import torch

from xanesnet.models import Model

from .base import Regularizer
from .registry import RegularizerRegistry


@RegularizerRegistry.register("l1")
class L1Reg(Regularizer):
    """L1 regularization (sum of absolute parameter values).

    Args:
        regularizer_type: Identifier string for this regularizer type.
        weight: Scalar multiplier applied to the L1 penalty.
    """

    def __init__(
        self,
        regularizer_type: str,
        weight: float,
    ) -> None:
        """Initialize ``L1Reg``."""
        super().__init__(regularizer_type, weight)

    def forward(self, model: Model) -> torch.Tensor:
        """Compute the weighted L1 penalty over all model parameters.

        Args:
            model: The model whose parameters are penalised.

        Returns:
            Scalar L1 regularization loss tensor.
        """
        params = torch.cat([parameter.reshape(-1) for parameter in model.parameters()])
        return params.abs().sum() * self.weight
