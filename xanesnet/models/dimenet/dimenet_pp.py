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

"""DimeNet++: fast directional message passing with reduced complexity."""

from collections.abc import Callable

import torch
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.utils import scatter

from ..registry import ModelRegistry
from .dimenet import DimeNet


@ModelRegistry.register("dimenet++")
class DimeNetPlusPlus(DimeNet):
    """DimeNet++ improvement over the original DimeNet.

    Reference: `"Fast and Uncertainty-Aware Directional Message Passing for
    Non-Equilibrium Molecules" <https://arxiv.org/abs/2011.14115>`_.

    Implementation based on the PyTorch Geometric reference:
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html

    Overrides the :class:`DimeNet` output and interaction blocks with the
    DimeNet++ variants. The interaction block replaces the bilinear tensor
    product with separate basis projections, and the output block adds an
    up-projection before the final per-atom head.

    Args:
        model_type: Model type string (passed to base class).
        hidden_channels: Hidden feature dimension.
        out_channels: Output feature dimension per atom.
        num_blocks: Number of interaction blocks.
        int_emb_size: Intermediate embedding size in the interaction block.
        basis_emb_size: Basis embedding size for RBF/SBF projection.
        out_emb_channels: Output embedding size before the final projection.
        num_spherical: Number of spherical basis functions; must be >= 2.
        num_radial: Number of radial basis functions.
        cutoff: Radial cutoff in **A**.
        envelope_exponent: Exponent controlling envelope smoothness.
        num_before_skip: Number of residual layers before the skip connection.
        num_after_skip: Number of residual layers after the skip connection.
        num_output_layers: Number of hidden layers in the output block.
        act: Activation function name.
        output_initializer: Weight init for the final layer; one of
            ``"zeros"`` or ``"glorot_orthogonal"``.
    """

    def __init__(
        self,
        model_type: str,
        # params:
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int,
        num_before_skip: int,
        num_after_skip: int,
        num_output_layers: int,
        act: str,
        output_initializer: str,
    ) -> None:
        super().__init__(
            model_type,
            hidden_channels,
            out_channels,
            num_blocks,
            1,  # num_bilinear is not used in DimeNet++
            num_spherical,
            num_radial,
            cutoff,
            envelope_exponent,
            num_before_skip,
            num_after_skip,
            num_output_layers,
            act,
            output_initializer,
        )

        self.int_emb_size = int_emb_size
        self.basis_emb_size = basis_emb_size
        self.out_emb_channels = out_emb_channels

        # Reuse the RBF, SBF, and embedding layers from DimeNet, then replace
        # the output and interaction blocks with their DimeNet++ variants.
        # The placeholder ``num_bilinear`` passed to ``super().__init__`` has
        # no effect here because it is only used by DimeNet's interaction
        # block, which we replace below.

        self.output_blocks = torch.nn.ModuleList(
            [
                OutputBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    self.act,
                    output_initializer,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    self.act,
                )
                for _ in range(num_blocks)
            ]
        )


class InteractionBlock(torch.nn.Module):
    """DimeNet++ interaction block.

    Replaces the bilinear weight tensor from DimeNet with two separate linear
    projections for RBF and SBF features, reducing memory and computation.

    Args:
        hidden_channels: Hidden feature dimension.
        int_emb_size: Intermediate triplet embedding size.
        basis_emb_size: Projection size for RBF and SBF basis features.
        num_spherical: Number of spherical basis functions.
        num_radial: Number of radial basis functions.
        num_before_skip: Number of residual layers before the skip connection.
        num_after_skip: Number of residual layers after the skip connection.
        act: Element-wise activation function.
    """

    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ) -> None:
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = torch.nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = torch.nn.Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = torch.nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = torch.nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = torch.nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = torch.nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = torch.nn.Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

    def reset_parameters(self) -> None:
        """Reset all learnable parameters to their initial distributions."""
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        sbf: torch.Tensor,
        idx_kj: torch.Tensor,
        idx_ji: torch.Tensor,
    ) -> torch.Tensor:
        """Compute updated message embeddings.

        Args:
            x: Message embeddings, shape ``(num_edges, hidden_channels)``.
            rbf: Radial basis features, shape ``(num_edges, num_radial)``.
            sbf: Spherical basis features,
                shape ``(num_triplets, num_spherical * num_radial)``.
            idx_kj: Index of the k->j edge for each triplet, shape ``(num_triplets,)``.
            idx_ji: Index of the j->i edge for each triplet, shape ``(num_triplets,)``.

        Returns:
            Updated message embeddings of shape ``(num_edges, hidden_channels)``.
        """
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce="sum")
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class ResidualLayer(torch.nn.Module):
    """Two-layer residual block with element-wise activation.

    Args:
        hidden_channels: Input and output feature dimension.
        act: Element-wise activation function.
    """

    def __init__(self, hidden_channels: int, act: Callable) -> None:
        super().__init__()
        self.act = act
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self) -> None:
        """Reset all learnable parameters to their initial distributions."""
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual transformation.

        Args:
            x: Input features of shape ``(*, hidden_channels)``.

        Returns:
            Output features of shape ``(*, hidden_channels)``.
        """
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class OutputBlock(torch.nn.Module):
    """DimeNet++ output block with an additional up-projection layer.

    Extends the DimeNet output block with an up-projection from
    ``hidden_channels`` to ``out_emb_channels`` before the final linear layers.

    Args:
        num_radial: Number of radial basis functions.
        hidden_channels: Hidden feature dimension.
        out_emb_channels: Output embedding dimension (up-projection target).
        out_channels: Output dimension per atom.
        num_layers: Number of hidden linear layers.
        act: Element-wise activation function.
        output_initializer: Weight init for the final layer; one of
            ``"zeros"`` or ``"glorot_orthogonal"``.
    """

    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = "zeros",
    ) -> None:
        assert output_initializer in {"zeros", "glorot_orthogonal"}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = torch.nn.Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = torch.nn.Linear(hidden_channels, out_emb_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(torch.nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = torch.nn.Linear(out_emb_channels, out_channels, bias=False)

    def reset_parameters(self) -> None:
        """Reset all learnable parameters to their initial distributions."""
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        self.lin_up.bias.data.fill_(0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_initializer == "zeros":
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == "glorot_orthogonal":
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        i: torch.Tensor,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        """Aggregate edge features and project to per-atom predictions.

        Args:
            x: Edge message embeddings, shape ``(num_edges, hidden_channels)``.
            rbf: Radial basis features, shape ``(num_edges, num_radial)``.
            i: Destination atom index for each edge, shape ``(num_edges,)``.
            num_nodes: Total number of atoms (used for scatter output size).

        Returns:
            Per-atom predictions of shape ``(num_nodes, out_channels)``.
        """
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)
