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

from torch import nn

from xanesnet.models.base_model import Model
from xanesnet.registry import ModelRegistry, SchemeRegistry
from xanesnet.utils.switch import ActivationSwitch


@ModelRegistry.register("aegan_mlp")
@SchemeRegistry.register("aegan_mlp", scheme_name="aegan")
class AEGAN_MLP(Model):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int = 256,
        num_hidden_layers_gen: int = 2,
        num_hidden_layers_shared: int = 2,
        num_hidden_layers_dis: int = 2,
        activation: str = "prelu",
    ):
        super().__init__()
        self.aegan_flag = 1

        # Save model configuration
        self.register_config(locals(), type="aegan_mlp")

        activation = ActivationSwitch().get(activation)

        # Initialise the generative autoencoder networks
        self.gen_a = Generator(
            input_size=in_size,
            num_hidden_layer=num_hidden_layers_gen,
            hidden_size=hidden_size,
            activation=activation,
        )  # generator for domain a
        self.gen_b = Generator(
            input_size=out_size,
            num_hidden_layer=num_hidden_layers_gen,
            hidden_size=hidden_size,
            activation=activation,
        )  # generator for domain b

        # Initialise the shared autoencoder layers
        self.shared_encoder = SharedLayer(
            num_hidden_layer=num_hidden_layers_shared,
            hidden_size=hidden_size,
            activation=activation,
        )
        self.shared_decoder = SharedLayer(
            num_hidden_layer=num_hidden_layers_shared,
            hidden_size=hidden_size,
            activation=activation,
        )

        # Initialise the discriminator networks
        self.dis_a = Discriminator(
            input_size=in_size,
            num_hidden_layer=num_hidden_layers_dis,
            hidden_size=hidden_size,
            activation=activation,
        )  # discriminator for domain a
        self.dis_b = Discriminator(
            input_size=out_size,
            num_hidden_layer=num_hidden_layers_dis,
            hidden_size=hidden_size,
            activation=activation,
        )  # discriminator for domain b

    # Reconstruct structure from structure
    def reconstruct_structure(self, x):
        enc = self.gen_a.encode(x)
        shared_enc = self.shared_encoder.forward(enc)
        shared_dec = self.shared_decoder.forward(shared_enc)
        recon = self.gen_a.decode(shared_dec)
        return recon

    # Reconstruct spectrum from spectrum
    def reconstruct_spectrum(self, x):
        enc = self.gen_b.encode(x)
        shared_enc = self.shared_encoder.forward(enc)
        shared_dec = self.shared_decoder.forward(shared_enc)
        recon = self.gen_b.decode(shared_dec)
        return recon

    # Predict spectrum from structure
    def predict_spectrum(self, x):
        enc = self.gen_a.encode(x)
        shared_enc = self.shared_encoder.forward(enc)
        shared_dec = self.shared_decoder.forward(shared_enc)
        pred = self.gen_b.decode(shared_dec)
        return pred

    # Predict structure from spectrum
    def predict_structure(self, x):
        enc = self.gen_b.encode(x)
        shared_enc = self.shared_encoder.forward(enc)
        shared_dec = self.shared_decoder.forward(shared_enc)
        pred = self.gen_a.decode(shared_dec)
        return pred

    # Reconstruct and predict spectrum and descriptor from inputs
    def generate_all(self, x_a, x_b):
        return self.forward(x_a, x_b)

    def forward(self, x_a, x_b):
        enc_a = self.gen_a.encode(x_a)
        enc_b = self.gen_b.encode(x_b)

        shared_enc_a = self.shared_encoder.forward(enc_a)
        shared_enc_b = self.shared_encoder.forward(enc_b)

        shared_dec_a = self.shared_decoder.forward(shared_enc_a)
        shared_dec_b = self.shared_decoder.forward(shared_enc_b)

        x_a_recon = self.gen_a.decode(shared_dec_a)
        x_b_recon = self.gen_b.decode(shared_dec_b)

        x_a_predict = self.gen_a.decode(shared_dec_b)
        x_b_predict = self.gen_b.decode(shared_dec_a)

        return x_a_recon, x_b_recon, x_a_predict, x_b_predict


class SharedLayer(nn.Module):
    # Shared Layer architecture
    def __init__(self, num_hidden_layer, hidden_size, activation):
        super().__init__()

        layers = []
        # Dense layers
        for _ in range(num_hidden_layer):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation)

        # Output layers
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dense_layers(x)
        return out


class Generator(nn.Module):
    # Generator architecture
    def __init__(self, input_size, num_hidden_layer, hidden_size, activation):
        super().__init__()

        encoder_layers = []
        # Encoder first Layer
        encoder_layers.append(nn.Linear(input_size, hidden_size))
        encoder_layers.append(nn.BatchNorm1d(hidden_size))
        encoder_layers.append(activation)

        # Encoder mid Layers
        for _ in range(num_hidden_layer - 1):
            encoder_layers.append(nn.Linear(hidden_size, hidden_size))
            encoder_layers.append(nn.BatchNorm1d(hidden_size))
            encoder_layers.append(activation)

        # Encoder final Layer
        encoder_layers.append(nn.Linear(hidden_size, hidden_size))
        encoder_layers.append(nn.BatchNorm1d(hidden_size))

        # Decoder input and mid layers
        decoder_layers = []
        # Decoder mid Layers
        for _ in range(num_hidden_layer):
            decoder_layers.append(nn.Linear(hidden_size, hidden_size))
            decoder_layers.append(nn.BatchNorm1d(hidden_size))
            decoder_layers.append(activation)

        # Decoder output layer
        decoder_layers.append(nn.Linear(hidden_size, input_size))

        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def encode(self, x):
        out = self.encoder_layers(x)
        return out

    def decode(self, x):
        out = self.decoder_layers(x)
        return out


class Discriminator(nn.Module):
    # Discriminator architecture
    def __init__(self, input_size, num_hidden_layer, hidden_size, activation):
        super().__init__()

        layers = []
        # First Layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(activation)

        # Mid Layers
        for _ in range(num_hidden_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation)

        # Final Layer
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dense_layers(x)
        return out
