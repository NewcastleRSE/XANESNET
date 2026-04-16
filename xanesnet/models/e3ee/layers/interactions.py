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

from typing import cast

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct

from .basic import CosineCutoff, IrrepNorm, RadialMLP


class EquivariantInteractionBlock(nn.Module):
    """
    Single equivariant message-passing interaction block using
    e3nn tensor products with gating.
    """

    def __init__(
        self,
        irreps_node: str,
        irreps_sh: str,
        irreps_message: str,
        rbf_dim: int,
        radial_hidden_dim: int,
        cutoff: float,
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        self.irreps_node = cast(o3.Irreps, o3.Irreps(irreps_node))
        self.irreps_sh = cast(o3.Irreps, o3.Irreps(irreps_sh))
        self.irreps_message = cast(o3.Irreps, o3.Irreps(irreps_message))

        self.pre_norm = IrrepNorm(self.irreps_node)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_message,
            shared_weights=False,
        )
        self.weight_mlp = RadialMLP(rbf_dim, radial_hidden_dim, self.tp.weight_numel)

        self.edge_gate = nn.Sequential(
            nn.Linear(rbf_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, 1),
            nn.Sigmoid(),
        )

        irreps_scalars = cast(o3.Irreps, o3.Irreps([(mul, ir) for mul, ir in self.irreps_message if ir.l == 0]))
        irreps_gated = cast(o3.Irreps, o3.Irreps([(mul, ir) for mul, ir in self.irreps_message if ir.l > 0]))
        irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e") if irreps_gated.num_irreps > 0 else o3.Irreps("")

        self.msg_linear = o3.Linear(
            self.irreps_message,
            irreps_scalars + irreps_gates + irreps_gated,
        )

        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[torch.nn.functional.silu] * len(irreps_scalars),
            irreps_gates=irreps_gates,
            act_gates=[torch.sigmoid] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )

        self.update_linear = o3.Linear(self.gate.irreps_out, self.irreps_node)
        self.self_linear = o3.Linear(self.irreps_node, self.irreps_node)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> torch.Tensor:
        if edge_src.numel() == 0:
            return x

        x_norm = self.pre_norm(x)

        tp_weights = self.weight_mlp(edge_rbf)
        m = self.tp(x_norm[edge_src], edge_sh, tp_weights)

        cutoff_w = self.cutoff_fn(edge_len).unsqueeze(-1)
        gate_w = self.edge_gate(edge_rbf)
        edge_w = cutoff_w * gate_w

        m = m * edge_w

        agg = torch.zeros(
            x.shape[0],
            self.irreps_message.dim,
            device=x.device,
            dtype=x.dtype,
        )
        agg.index_add_(0, edge_dst, m)

        norm = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        norm.index_add_(0, edge_dst, edge_w)
        agg = agg / norm.clamp_min(1e-8)

        agg = self.msg_linear(agg)
        agg = self.gate(agg)

        out = self.self_linear(x_norm) + self.update_linear(agg)
        return x + self.res_scale.to(x.dtype) * out
