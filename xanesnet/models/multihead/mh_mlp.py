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

"""Multi-head MLP model for spectroscopy and descriptor prediction."""

import torch
from torch import nn

from xanesnet.components import ActivationRegistry, BiasInitRegistry, WeightInitRegistry
from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry
from .layers import MLPHead


@ModelRegistry.register("mh_mlp")
class MultiHeadMLP(Model):
    """Multi-head MLP for forward or inverse prediction.

    A shared MLP trunk produces features for a collection of independent MLP
    prediction heads. The heads are evaluated together during a forward pass;
    the multi-head batch processor selects the head associated with each sample.
    The dataset and batch-processor direction determines whether the model
    maps descriptors to spectra or spectra to descriptors.

    Args:
        model_type: Model type identifier string.
        in_size: Number of input features.
        out_size: Number of output features for each prediction head. All
            heads must have the same output size so their predictions can be
            stacked into one tensor.
        hidden_size: Width of the first shared hidden layer.
        dropout: Dropout probability applied in shared and head hidden layers.
        num_hidden_layers: Number of shared hidden layers.
        shrink_rate: Multiplicative factor applied to shared layer widths.
        activation: Name of the activation function.
        head_num_hidden_layers: Number of hidden layers in each prediction head.
        head_hidden_size: Width of the first hidden layer in each head.
        head_shrink_rate: Multiplicative factor applied to head layer widths.
    """

    def __init__(
        self,
        model_type: str,
        # params:
        in_size: int,
        out_size: list[int],
        hidden_size: int,
        dropout: float,
        num_hidden_layers: int,
        shrink_rate: float,
        activation: str,
        head_num_hidden_layers: int,
        head_hidden_size: int,
        head_shrink_rate: float,
    ) -> None:
        """Initialize the shared trunk and per-head MLP predictors."""
        super().__init__(model_type)

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.shrink_rate = shrink_rate
        self.activation = activation
        self.head_num_hidden_layers = head_num_hidden_layers
        self.head_hidden_size = head_hidden_size
        self.head_shrink_rate = head_shrink_rate

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

        # Initialise dense layers
        self.dense_layers = nn.Sequential(*layers)

        # Initialise heads
        self.heads = nn.ModuleList(
            [
                MLPHead(
                    in_size=current_size,
                    out_size=out,
                    num_hidden_layers=head_num_hidden_layers,
                    hidden_size=head_hidden_size,
                    shrink_rate=head_shrink_rate,
                    dropout=dropout,
                    activation=activation,
                )
                for out in out_size
            ]
        )

    def forward(self, x: torch.Tensor, active_head_idx: int | None = None) -> torch.Tensor:
        """Run a forward pass through the Multi-head MLP.

        Args:
            x: Input tensor with shape ``(batch_size, in_size)``.
            active_head_idx: Optional index of one head. When ``None``, return
                predictions from every head.

        Returns:
            If ``active_head_idx`` is ``None``, a tensor with shape
            ``(num_heads, batch_size, out_size)``; otherwise a tensor with
            shape ``(batch_size, out_size)``.
        """
        shared = self.dense_layers(x)
        if active_head_idx is None:
            return torch.stack([head(shared) for head in self.heads], dim=0)
        else:
            return self.heads[active_head_idx](shared)

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """Initialize all linear layer weights and biases.

        Args:
            weights_init: Name of the weight initialization scheme (looked up via
                ``WeightInitRegistry``).
            bias_init: Name of the bias initialization scheme (looked up via
                ``BiasInitRegistry``).
            **kwargs: Extra keyword arguments forwarded to the weight initializer.
        """
        weight_init_fn = WeightInitRegistry.get(weights_init)
        bias_init_fn = BiasInitRegistry.get(bias_init)

        def _init_layer(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                weight_init_fn(m.weight, **kwargs)
                assert m.bias is not None, "Bias is None, cannot initialize."
                bias_init_fn(m.bias)

        # Apply to all modules
        self.apply(_init_layer)

    @property
    def signature(self) -> Config:
        """Return the model signature.

        Returns:
            Configuration values needed to recreate this model.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "in_size": self.in_size,
                "out_size": self.out_size,
                "hidden_size": self.hidden_size,
                "dropout": self.dropout,
                "num_hidden_layers": self.num_hidden_layers,
                "shrink_rate": self.shrink_rate,
                "activation": self.activation,
                "head_num_hidden_layers": self.head_num_hidden_layers,
                "head_hidden_size": self.head_hidden_size,
                "head_shrink_rate": self.head_shrink_rate,
            }
        )
        return signature
