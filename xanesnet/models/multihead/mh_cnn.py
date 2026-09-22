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

"""Multi-head one-dimensional convolutional model for spectroscopy and descriptor prediction."""

import torch
from torch import nn

from xanesnet.components import ActivationRegistry, BiasInitRegistry, WeightInitRegistry
from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry
from .layers import MLPHead


@ModelRegistry.register("mh_cnn")
class MultiHeadCNN(Model):
    """Multi-head 1D CNN for forward or inverse prediction.

    A shared convolutional encoder produces features for a collection of
    independent MLP prediction heads. The heads are evaluated together during
    a forward pass; the multi-head batch processor selects the head associated
    with each sample. The dataset and batch-processor direction determines
    whether the model maps descriptors to spectra or spectra to descriptors.

    Args:
        model_type: Model type identifier string.
        in_size: Number of input features.
        out_size: Number of output features for each prediction head. All
            heads must have the same output size so their predictions can be
            stacked into one tensor.
        dropout: Dropout probability applied in convolutional and head hidden layers.
        num_conv_layers: Number of convolutional layers in the shared encoder.
        activation: Name of the activation function.
        out_channel: Number of output channels in the first convolutional layer.
        channel_mul: Multiplicative channel-width factor between convolutional layers.
        kernel_size: Size of each convolutional kernel.
        stride: Stride used by each convolutional layer.
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
        dropout: float,
        num_conv_layers: int,
        activation: str,
        out_channel: int,
        channel_mul: int,
        kernel_size: int,
        stride: int,
        head_num_hidden_layers: int,
        head_hidden_size: int,
        head_shrink_rate: float,
    ) -> None:
        """Initialize the shared convolutional encoder and prediction heads."""
        super().__init__(model_type)

        self.in_size = in_size
        self.out_size = out_size
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        self.activation = activation
        self.out_channel = out_channel
        self.channel_mul = channel_mul
        self.kernel_size = kernel_size
        self.stride = stride
        self.head_num_hidden_layers = head_num_hidden_layers
        self.head_hidden_size = head_hidden_size
        self.head_shrink_rate = head_shrink_rate

        conv_layers: list[nn.Module] = []

        # Initialise convolutional layers
        in_channel = 1
        current_out_channel = out_channel
        for _ in range(num_conv_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, current_out_channel, kernel_size, stride),
                    nn.BatchNorm1d(current_out_channel),
                    ActivationRegistry.create(activation),
                    nn.Dropout(p=dropout),
                )
            )
            in_channel = current_out_channel
            current_out_channel *= channel_mul

        self.conv_layers = nn.Sequential(*conv_layers)
        conv_out_size = self._get_conv_output_size(in_size)

        # Initialise multi-headed predictor
        self.heads = nn.ModuleList(
            [
                MLPHead(
                    in_size=conv_out_size,
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

    def _get_conv_output_size(self, in_size: int) -> int:
        """Calculate the flattened feature dimension of the convolutional encoder.

        The temporary evaluation mode prevents the dummy batch from updating
        batch-normalization statistics during model construction.

        Args:
            in_size: Length of the one-dimensional input feature sequence.

        Returns:
            Flattened feature count produced by the convolutional encoder.
        """
        dummy_input = torch.randn(1, 1, in_size)
        was_training = self.conv_layers.training
        self.conv_layers.eval()
        try:
            with torch.no_grad():
                output = self.conv_layers(dummy_input)
        finally:
            self.conv_layers.train(was_training)

        return int(output[0].numel())

    def forward(self, x: torch.Tensor, active_head_idx: int | None = None) -> torch.Tensor:
        """Run a forward pass through the multi-head CNN.

        Args:
            x: Input tensor with shape ``(batch_size, in_size)``.
            active_head_idx: Optional index of one head. When ``None``, return
                predictions from every head.

        Returns:
            If ``active_head_idx`` is ``None``, a tensor with shape
            ``(num_heads, batch_size, out_size)``; otherwise a tensor with
            shape ``(batch_size, out_size)``.
        """
        x = x.unsqueeze(1)
        shared = self.conv_layers(x)
        shared = torch.flatten(shared, 1)

        if active_head_idx is None:
            return torch.stack([head(shared) for head in self.heads], dim=0)
        else:
            return self.heads[active_head_idx](shared)

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """Initialize all convolutional and linear layer weights and biases.

        Args:
            weights_init: Name of the weight initialization scheme.
            bias_init: Name of the bias initialization scheme.
            **kwargs: Extra keyword arguments forwarded to the weight initializer.
        """
        weight_init_fn = WeightInitRegistry.get(weights_init)
        bias_init_fn = BiasInitRegistry.get(bias_init)

        def _init_layer(m: nn.Module) -> None:
            """Initialize one supported trainable layer in place."""
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
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
                "dropout": self.dropout,
                "num_conv_layers": self.num_conv_layers,
                "activation": self.activation,
                "out_channel": self.out_channel,
                "channel_mul": self.channel_mul,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "head_num_hidden_layers": self.head_num_hidden_layers,
                "head_hidden_size": self.head_hidden_size,
                "head_shrink_rate": self.head_shrink_rate,
            }
        )
        return signature
