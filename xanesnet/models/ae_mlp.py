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


@register_model("ae_mlp")
@register_scheme("ae_mlp", scheme_name="ae")
class AE_MLP(Model):
    """
    A class for constructing an AE-MLP (Autoencoder Multilayer Perceptron Network).

    The model consists of three modular components:
    1. An Encoder that compresses the input to a latent representation.
    2. A Decoder that reconstructs the input from the latent representation.
    3. A Predictor ('dense_layers') that predicts the output from the latent representation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int = 256,
        dropout: float = 0.2,
        num_hidden_layers: int = 3,
        shrink_rate: float = 1.0,
        activation: str = "prelu",
    ):
        """
        Args:
            in_features (integer): Size of input data
            out_features (integer): Size of output data
            hidden_size (integer): Size of the initial hidden layer.
            dropout (float): Dropout probability for hidden layers.
            num_hidden_layers (int): Number of hidden layers in the encoder.
            shrink_rate (float): Rate to reduce the hidden layer size multiplicatively.
            activation (str): Name of activation function for hidden layers.
        """
        super().__init__()
        self.ae_flag = 1

        # Save model configuration
        self.register_config(locals(), type="ae_mlp")

        act_fn = ActivationSwitch().get(activation)

        # --- Encoder Construction ---
        enc_layers = []
        current_size = in_features
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate**i))
            if next_size < 1:
                raise ValueError(f"Encoder layer {i + 1} size is less than 1.")

            enc_layers.append(nn.Linear(current_size, next_size))
            enc_layers.append(act_fn)
            current_size = next_size

        # Final latent layer
        latent_size = int(hidden_size * (shrink_rate**num_hidden_layers))
        if latent_size < 1:
            raise ValueError("Latent size is less than 1.")
        enc_layers.append(nn.Linear(current_size, latent_size))

        self.encoder_layers = nn.Sequential(*enc_layers)

        # --- Decoder Construction ---
        dec_layers = []
        current_size = latent_size
        # Create a list of the encoder layer dimensions in reverse
        layer_sizes = [in_features] + [
            int(hidden_size * (shrink_rate**i)) for i in range(num_hidden_layers)
        ]
        for size in reversed(layer_sizes):
            dec_layers.append(nn.Linear(current_size, size))
            dec_layers.append(act_fn)
            current_size = size

        # Remove the last activation layer
        dec_layers.pop()
        self.decoder_layers = nn.Sequential(*dec_layers)

        # --- Predictor Construction ---
        fc_layers = []
        # Input to predictor is the latent space
        fc_layers.append(nn.Linear(latent_size, hidden_size))
        fc_layers.append(act_fn)
        fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(hidden_size, out_features))

        self.dense_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Feed forward through dense layers
        out = self.encoder_layers(x)
        predict = self.dense_layers(out)
        reconstruct = self.decoder_layers(out)

        return reconstruct, predict

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # Generate predictions based on the encoded representation of the input.
        out = self.encoder_layers(x)
        pred = self.dense_layers(out)

        return pred

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct the input data based on the encoded representation.
        out = self.encoder_layers(x)
        recon = self.decoder_layers(out)

        return recon
