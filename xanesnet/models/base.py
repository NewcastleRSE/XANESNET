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

"""Abstract base class for all XANESNET models."""

from abc import ABC, abstractmethod

from torch import nn

from xanesnet.serialization.config import Config

###############################################################################
#################################### CLASS ####################################
###############################################################################


class Model(nn.Module, ABC):
    """Abstract base class for all XANESNET models.

    Subclasses must implement ``init_weights`` and ``signature``.

    Args:
        model_type: String identifier for the model type.
    """

    def __init__(
        self,
        model_type: str,
    ) -> None:
        """Initialize ``Model``."""
        super().__init__()

        self.model_type = model_type

    @abstractmethod
    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """Initialize all model weights and biases.

        Args:
            weights_init: Name of the weight initialization scheme.
            bias_init: Name of the bias initialization scheme.
            **kwargs: Extra keyword arguments forwarded to the weight initializer.
        """
        ...

    @property
    @abstractmethod
    def signature(self) -> Config:
        """Return the model signature.

        Returns:
            Configuration values needed to recreate this model.
        """
        signature = Config(
            {
                "model_type": self.model_type,
            }
        )
        return signature
