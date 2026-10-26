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

"""Reusable MLP prediction head for multi-head models."""

import torch
from torch import nn

from xanesnet.components import ActivationRegistry


class MLPHead(nn.Module):
    """MLP prediction head used by the multi-head model architectures.

    Each hidden layer contains a linear transformation, dropout, and an
    activation function. The output layer is linear so the head can predict
    both encoded spectra and descriptor features in forward or inverse mode.

    Args:
        in_size: Number of input features.
        out_size: Number of output features.
        hidden_size: Width of the first hidden layer.
        dropout: Dropout probability applied after each hidden layer.
        num_hidden_layers: Number of hidden layers.
        shrink_rate: Multiplicative factor applied to hidden layer widths.
        activation: Name of the hidden-layer activation function.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        dropout: float,
        num_hidden_layers: int,
        shrink_rate: float,
        activation: str,
    ) -> None:
        """Initialize ``MLPHead``."""
        super().__init__()

        layers: list[nn.Module] = []

        # Initialise input and hidden layers
        current_size = in_size
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate**i))
            if next_size < 1:
                raise ValueError(f"Hidden layer {i + 1} size is less than 1. Adjust hidden_size or shrink_rate.")

            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.Dropout(dropout))
            layers.append(ActivationRegistry.create(activation))
            current_size = next_size

        # Initialise output layer
        layers.append(nn.Linear(current_size, out_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the head prediction for ``x``.

        Args:
            x: Input tensor. ``(batch_size, in_size)``.

        Returns:
            Prediction tensor. ``(batch_size, out_size)``.
        """
        return self.model(x)
