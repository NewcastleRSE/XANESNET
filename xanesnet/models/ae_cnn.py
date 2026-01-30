"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch import nn

from xanesnet.models.base_model import Model
from xanesnet.registry import register_model, register_scheme
from xanesnet.utils.switch import ActivationSwitch


@register_model("ae_cnn")
@register_scheme("ae_cnn", scheme_name="ae")
class AE_CNN(Model):
    """
    A class for constructing a AE-CNN (Autoencoder Convolutional Neural Network) model.
    The model has three main components: encoder, decoder and dense layers.
    The reconstruction of input data is performed as a forward pass through the encoder
    and decoder. The prediction is performed as a forward pass through the encoder and
    dense layers. Hyperparameter specification is the same as for the CNN model type.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_channel: int = 32,
        channel_mul: int = 2,
        hidden_size: int = 64,
        dropout: float = 0.2,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "prelu",
        num_conv_layers: int = 3,
    ):
        """
        Args:
            in_features (int): Size of input data (length of the 1D signal).
            out_features (int): Size of output data.
            hidden_size (int): Size of the hidden layer in the dense predictor.
            dropout (float): Dropout rate for regularization.
            num_conv_layers (int): Number of convolutional layers in the encoder.
            activation (str): Name of activation function for all layers.
            out_channel (int): Number of output channels for the first conv layer.
            channel_mul (int): Multiplies the number of channels at each subsequent layer.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for convolution and upsampling.
        """
        super().__init__()
        self.ae_flag = 1

        # Save model configuration
        self.register_config(locals(), type="ae_cnn")

        act_fn = ActivationSwitch().get(activation)

        # Start collecting shape of convolutional layers for calculating padding
        all_conv_shapes = [in_features]

        # Starting shape
        conv_shape = in_features

        # Construct encoder convolutional layers
        enc_layers = []
        enc_in_channel = 1
        enc_out_channel = out_channel
        for block in range(num_conv_layers):
            # Create conv layer
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=enc_in_channel,
                    out_channels=enc_out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                act_fn,
            )

            enc_layers.append(conv_layer)

            # Update in and out channels
            enc_in_channel = enc_out_channel
            enc_out_channel = enc_out_channel * channel_mul

            # Update output shape for conv layer
            conv_shape = int(((conv_shape - kernel_size) / stride) + 1)
            all_conv_shapes.append(conv_shape)

        self.encoder_layers = nn.Sequential(*enc_layers)

        # Construct predictor dense layers
        dense_in_shape = (
            out_channel * channel_mul ** (num_conv_layers - 1) * all_conv_shapes[-1]
        )

        dense_layers = []

        dense_layer1 = nn.Sequential(
            nn.Linear(dense_in_shape, hidden_size),
            act_fn,
            nn.Dropout(dropout),
        )

        dense_layer2 = nn.Sequential(
            nn.Linear(hidden_size, out_features),
        )

        dense_layers.append(dense_layer1)
        dense_layers.append(dense_layer2)

        self.dense_layers = nn.Sequential(*dense_layers)

        # Construct decoder transpose convolutional layers
        dec_in_channel = out_channel * channel_mul ** (num_conv_layers - 1)
        dec_out_channel = out_channel * channel_mul ** (num_conv_layers - 2)

        dec_layers = []

        for block in range(num_conv_layers):
            tconv_out_shape = all_conv_shapes[num_conv_layers - block - 1]
            tconv_in_shape = all_conv_shapes[num_conv_layers - block]

            tconv_shape = int(((tconv_in_shape - 1) * stride) + kernel_size)

            # Calculate padding to input or output of transpose conv layer
            if tconv_shape != tconv_out_shape:
                if tconv_shape > tconv_out_shape:
                    # Pad input to transpose conv layer
                    padding = tconv_shape - tconv_out_shape
                    output_padding = 0
                elif tconv_shape < tconv_out_shape:
                    # Pad output of transpose conv layer
                    padding = 0
                    output_padding = tconv_out_shape - tconv_shape
            else:
                padding = 0
                output_padding = 0

            if block == num_conv_layers - 1:
                dec_out_channel = 1

            tconv_layer = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=dec_in_channel,
                    out_channels=dec_out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    output_padding=output_padding,
                    padding=padding,
                ),
                act_fn,
            )
            dec_layers.append(tconv_layer)

            # Update in/out channels
            if block < num_conv_layers - 1:
                dec_in_channel = dec_out_channel
                dec_out_channel = dec_out_channel // channel_mul

        self.decoder_layers = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)

        out = self.encoder_layers(x)

        pred = out.view(out.size(0), -1)
        pred = self.dense_layers(pred)

        recon = self.decoder_layers(out)
        recon = recon.squeeze(dim=1)

        return recon, pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # Generate predictions based on the encoded representation of the input.
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)

        out = self.encoder_layers(x)

        pred = out.view(out.size(0), -1)
        pred = self.dense_layers(pred)

        return pred

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct the input data based on the encoded representation.
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)

        out = self.encoder_layers(x)

        recon = self.decoder_layers(out)
        recon = recon.squeeze(dim=1)

        return recon
