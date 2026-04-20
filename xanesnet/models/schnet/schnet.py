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

import math

import torch
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.typing import OptTensor

from xanesnet.components import BiasInitRegistry, WeightInitRegistry
from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry


@ModelRegistry.register("schnet")
class SchNet(Model):
    """
    The continuous-filter convolutional neural network SchNet:
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions"`;
    Arxiv: `<https://arxiv.org/abs/1706.08566>`;
    Implementation similar to `<https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/nn/models/schnet.html>`;
    Adapted for XANES prediction.
    """

    def __init__(
        self,
        model_type: str,
        # params:
        hidden_channels: int,
        reduce_channels_1: int,
        reduce_channels_2: int,
        num_filters: int,
        num_interactions: int,
        num_gaussians: int,
        cutoff: float,
        mean_spectrum: list[float] | None,
    ) -> None:
        super().__init__(model_type)

        self.hidden_channels = hidden_channels
        self.reduce_channels_1 = reduce_channels_1
        self.reduce_channels_2 = reduce_channels_2
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.mean_spectrum = mean_spectrum

        # Mean spectrum for residual learning
        if mean_spectrum is not None:
            self.register_buffer("mean_tensor", torch.tensor(mean_spectrum, dtype=torch.float32))

        # Support z == 0 for padding atoms so that their embedding vectors
        # are zeroed and do not receive any gradients.
        self.embedding = torch.nn.Embedding(100, hidden_channels, padding_idx=0)

        # Continuous vector encoding for scalar distance values using Gaussian functions
        self.distance_encoding = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Defining the network layers

        # Interaction layers
        self.interactions = torch.nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)

        # Linear layer
        self.lin1 = torch.nn.Linear(hidden_channels, reduce_channels_1)
        self.act = ShiftedSoftplus()
        self.lin2 = torch.nn.Linear(reduce_channels_1, reduce_channels_2)

    def forward(
        self,
        z: torch.Tensor,  # Atomic number of each atom with shape [num_atoms].
        edge_index: torch.Tensor,  # Edge indices with shape [2, num_edges].
        edge_weight: torch.Tensor,  # Edge weights (interatomic distances) with shape [num_edges].
        batch: OptTensor = None,  # Batch indices assigning each atom to a separate molecule with shape [num_atoms].
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        # Create 0 batch tensor if batch is not given
        batch = torch.zeros_like(z) if batch is None else batch

        # Atomic number embeddings
        h = self.embedding(z)

        # Scalar distance encoding
        edge_attr = self.distance_encoding(edge_weight)

        # Interaction layers
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Linear layers
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.mean_spectrum is not None:
            h = h + self.mean_tensor  # Only learning the residual to the mean spectrum

        # Return one output vector per site across the whole batch.
        return h

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        # Init embedding parameters (non-linear layer, keep default init)
        self.embedding.reset_parameters()

        # Init interaction blocks (includes CFConv, MLP sub-networks)
        for interaction in self.interactions:
            interaction.init_weights(weights_init, bias_init, **kwargs)

        # Init top-level linear layers
        weight_init_fn = WeightInitRegistry.get(weights_init, **kwargs)
        bias_init_fn = BiasInitRegistry.get(bias_init)

        weight_init_fn(self.lin1.weight)
        bias_init_fn(self.lin1.bias)
        weight_init_fn(self.lin2.weight)
        bias_init_fn(self.lin2.bias)

    @property
    def signature(self) -> Config:
        """
        Return model signature as a dictionary.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "hidden_channels": self.hidden_channels,
                "reduce_channels_1": self.reduce_channels_1,
                "reduce_channels_2": self.reduce_channels_2,
                "num_filters": self.num_filters,
                "num_interactions": self.num_interactions,
                "num_gaussians": self.num_gaussians,
                "cutoff": self.cutoff,
                "mean_spectrum": self.mean_spectrum,
            }
        )
        return signature


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float,
    ) -> None:
        super().__init__()

        # Shallow MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            torch.nn.Linear(num_filters, num_filters),
        )

        # Continuous Filter Convolution
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)

        # Activation Function
        self.act = ShiftedSoftplus()

        # Single linear layer
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        weight_init_fn = WeightInitRegistry.get(weights_init, **kwargs)
        bias_init_fn = BiasInitRegistry.get(bias_init)

        weight_init_fn(self.mlp[0].weight)
        bias_init_fn(self.mlp[0].bias)
        weight_init_fn(self.mlp[2].weight)
        bias_init_fn(self.mlp[2].bias)
        self.conv.init_weights(weights_init, bias_init, **kwargs)
        weight_init_fn(self.lin.weight)
        bias_init_fn(self.lin.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(tgnn.MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        net: torch.nn.Sequential,
        cutoff: float,
    ) -> None:
        super().__init__(aggr="add")
        self.lin1 = torch.nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = torch.nn.Linear(num_filters, out_channels)
        # E.g.: shallow MLP
        self.net = net
        self.cutoff = cutoff

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        weight_init_fn = WeightInitRegistry.get(weights_init, **kwargs)
        bias_init_fn = BiasInitRegistry.get(bias_init)

        weight_init_fn(self.lin1.weight)
        weight_init_fn(self.lin2.weight)
        bias_init_fn(self.lin2.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        C = 0.5 * (torch.cos(edge_weight * math.pi / self.cutoff) + 1.0)
        W = self.net(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - self.shift
