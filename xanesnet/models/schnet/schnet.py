# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""SchNet continuous-filter convolutional neural network for XANES prediction."""

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
    """SchNet continuous-filter convolutional neural network for molecular property prediction.

    Adapted for XANES spectral prediction. Architecture follows the original SchNet paper with
    interaction blocks built from continuous-filter convolutions (CFConv) and ShiftedSoftplus
    activations.

    Note:
        Paper: "SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum
        Interactions" (https://arxiv.org/abs/1706.08566). Implementation adapted from PyTorch
        Geometric's SchNet (https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/
        torch_geometric/nn/models/schnet.html).

    Args:
        model_type: Model type identifier string.
        hidden_channels: Atom embedding dimension used throughout all interaction blocks.
        reduce_channels_1: Output dimension of the first post-interaction linear layer.
        reduce_channels_2: Output dimension of the second post-interaction linear layer
            (equals the spectral output size).
        num_filters: Number of filters in the CFConv layers and filter MLP.
        num_interactions: Number of interaction blocks.
        num_gaussians: Number of Gaussian basis functions for distance encoding.
        cutoff: Radial cutoff distance. **Angstrom**.
        mean_spectrum: Optional reference mean spectrum for residual learning. If provided,
            it is added back on every forward pass after the output head. Its length must
            equal ``reduce_channels_2``. ``(reduce_channels_2,)``
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
        """Initialize ``SchNet``."""
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
            if len(mean_spectrum) != reduce_channels_2:
                raise ValueError(
                    "mean_spectrum length must match reduce_channels_2 "
                    f"({len(mean_spectrum)} != {reduce_channels_2})"
                )
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
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: OptTensor = None,  # TODO we might be able to remove this
    ) -> torch.Tensor:
        """Run a forward pass through SchNet.

        Args:
            z: Atomic numbers. ``(num_atoms,)``
            edge_index: Edge indices (source, target). ``(2, num_edges)``
            edge_weight: Interatomic distances. ``(num_edges,)`` **Angstrom**.
            batch: Batch assignment indices mapping each atom to a graph. ``(num_atoms,)``
                Accepted for interface compatibility with batched PyG models, but ignored by
                this node-level SchNet variant because it returns per-atom outputs.

        Returns:
            Per-atom output vectors. ``(num_atoms, reduce_channels_2)``
        """
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
            h = h + self.mean_tensor  # Residual learning: add back the mean spectrum.

        return h

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """Initialize all learnable layer weights and biases.

        Args:
            weights_init: Weight initialization scheme name (looked up via
                ``WeightInitRegistry``).
            bias_init: Bias initialization scheme name (looked up via ``BiasInitRegistry``).
            **kwargs: Extra keyword arguments forwarded to the weight initializer.
        """
        # Embedding uses default initialization (non-linear lookup layer).
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
        """Return the model signature as a :class:`~xanesnet.serialization.config.Config`."""
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
    """Single SchNet interaction block.

    Applies a continuous-filter convolution (CFConv) followed by a ShiftedSoftplus
    activation and a linear projection.

    Args:
        hidden_channels: Atom embedding and hidden feature dimension.
        num_gaussians: Number of Gaussian basis functions (input width to the filter MLP).
        num_filters: Number of filters in the CFConv and filter MLP layers.
        cutoff: Radial cutoff distance. **Angstrom**.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float,
    ) -> None:
        """Initialize ``InteractionBlock``."""
        super().__init__()

        # Shallow filter MLP: maps Gaussian-smeared distances to per-edge filter weights.
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
        """Initialize weights and biases for all sub-layers.

        Args:
            weights_init: Weight initialization scheme name.
            bias_init: Bias initialization scheme name.
            **kwargs: Extra keyword arguments forwarded to the weight initializer.
        """
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
        """Run the interaction block forward pass.

        Args:
            x: Atom feature matrix. ``(num_atoms, hidden_channels)``
            edge_index: Edge indices. ``(2, num_edges)``
            edge_weight: Interatomic distances. ``(num_edges,)`` **Angstrom**.
            edge_attr: Gaussian-smeared distances. ``(num_edges, num_gaussians)``

        Returns:
            Updated atom features. ``(num_atoms, hidden_channels)``
        """
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(tgnn.MessagePassing):
    """Continuous-filter convolutional layer (message-passing variant).

    Applies a filter network to edge attributes to produce per-edge weights, multiplied
    by a cosine envelope, and aggregates weighted neighbor features via sum pooling.

    Args:
        in_channels: Input node feature dimension.
        out_channels: Output node feature dimension.
        num_filters: Intermediate filter dimension used by both linear projections.
        net: Filter network mapping Gaussian-smeared distances to per-edge filter weights.
        cutoff: Radial cutoff distance for the cosine envelope. **Angstrom**.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        net: torch.nn.Sequential,
        cutoff: float,
    ) -> None:
        """Initialize ``CFConv``."""
        super().__init__(aggr="add")
        self.lin1 = torch.nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = torch.nn.Linear(num_filters, out_channels)
        self.net = net
        self.cutoff = cutoff

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """Initialize linear layer weights and biases.

        Args:
            weights_init: Weight initialization scheme name.
            bias_init: Bias initialization scheme name.
            **kwargs: Extra keyword arguments forwarded to the weight initializer.
        """
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
        """Run the CFConv forward pass.

        Args:
            x: Node features. ``(num_nodes, in_channels)``
            edge_index: Edge indices. ``(2, num_edges)``
            edge_weight: Interatomic distances. ``(num_edges,)`` **Angstrom**.
            edge_attr: Gaussian-smeared distances. ``(num_edges, num_gaussians)``

        Returns:
            Updated node features. ``(num_nodes, out_channels)``
        """
        C = 0.5 * (torch.cos(edge_weight * math.pi / self.cutoff) + 1.0)
        W = self.net(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute messages as element-wise product of neighbor features and filter weights.

        Args:
            x_j: Source node features. ``(num_edges, num_filters)``
            W: Per-edge filter weights. ``(num_edges, num_filters)``

        Returns:
            Messages. ``(num_edges, num_filters)``
        """
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    """Gaussian smearing layer for encoding scalar distances.

    Expands scalar distance values into a vector of Gaussian basis function evaluations
    evenly spaced between ``start`` and ``stop``.

    Args:
        start: Center of the first Gaussian basis function. **Angstrom**.
        stop: Center of the last Gaussian basis function. **Angstrom**.
        num_gaussians: Number of evenly-spaced Gaussian basis functions. Must be at least 2.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ) -> None:
        """Initialize ``GaussianSmearing``."""
        super().__init__()
        if num_gaussians < 2:
            raise ValueError(f"num_gaussians must be at least 2, got {num_gaussians}")
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Expand distances into Gaussian basis representation.

        Args:
            dist: Interatomic distances. ``(num_edges,)`` **Angstrom**.

        Returns:
            Gaussian basis encodings. ``(num_edges, num_gaussians)``
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    """Shifted Softplus activation function.

    Computes ``softplus(x) - log(2)`` so that the output is zero at ``x = 0``,
    matching the shift used in the SchNet paper.
    """

    def __init__(self) -> None:
        """Initialize ``ShiftedSoftplus``."""
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shifted softplus activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor with the same shape as ``x``.
        """
        return F.softplus(x) - self.shift
