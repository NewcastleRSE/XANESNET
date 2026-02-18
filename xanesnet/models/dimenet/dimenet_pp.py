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

from collections.abc import Callable

import torch
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.utils import scatter

from ..registry import ModelRegistry
from .dimenet import DimeNet


@ModelRegistry.register("dimenet++")
class DimeNetPlusPlus(DimeNet):
    """The DimeNet++ improvement of the original DimeNet:
    `"Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules"`;
    Arxiv: `<https://arxiv.org/abs/2011.14115>`;
    Implementation similar to `<https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html>`
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
        max_num_neighbors: int,
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
            max_num_neighbors,
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

        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initalization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:

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

    def __init__(self, hidden_channels: int, act: Callable) -> None:
        super().__init__()
        self.act = act
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self) -> None:
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class OutputBlock(torch.nn.Module):

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
        self.lin_up = torch.nn.Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(torch.nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = torch.nn.Linear(out_emb_channels, out_channels, bias=False)

    def reset_parameters(self) -> None:
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
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
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)
