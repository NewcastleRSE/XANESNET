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

"""Branch fusion layers for E3EE."""

import torch
import torch.nn as nn

from .basic import MLP


class GatedBranchFusion(nn.Module):
    """Energy-conditioned gated fusion of active branch latents.

    Each active branch is projected to ``fused_dim`` and combined with a
    learned soft gate predicted from the projected branch values, their
    magnitudes, and a conditioning feature such as the energy embedding.

    Args:
        branch_dims: Feature dimensions of the active branch tensors.
        fused_dim: Common latent dimension used for gated summation.
        cond_dim: Dimension of the conditioning features.
        hidden_dim: Hidden dimension of the gate and output MLPs.
        use_softmax: If ``True``, normalize gates with softmax. Otherwise,
            use sigmoid gates normalized to sum to one.
    """

    def __init__(
        self,
        branch_dims: list[int],
        fused_dim: int,
        cond_dim: int,
        hidden_dim: int,
        use_softmax: bool = True,
    ) -> None:
        """Initialize ``GatedBranchFusion``."""
        super().__init__()
        if len(branch_dims) == 0:
            raise ValueError("GatedBranchFusion requires at least one branch")

        self.n_branches = len(branch_dims)
        self.fused_dim = fused_dim
        self.use_softmax = use_softmax

        self.proj = nn.ModuleList(
            [nn.Identity() if d == fused_dim else nn.Linear(d, fused_dim) for d in branch_dims]
        )

        gate_in_dim = self.n_branches * (2 * fused_dim) + cond_dim
        self.gate_mlp = MLP(
            in_dim=gate_in_dim,
            hidden_dim=hidden_dim,
            out_dim=self.n_branches,
            n_layers=3,
        )

        self.out_mlp = MLP(
            in_dim=fused_dim,
            hidden_dim=hidden_dim,
            out_dim=fused_dim,
            n_layers=2,
        )

        self.reset_gate_logits()

    def reset_gate_logits(self) -> None:
        """Initialize the final gate layer to produce uniform gates."""
        last_linear = None
        for module in reversed(self.gate_mlp.net):
            if isinstance(module, nn.Linear):
                last_linear = module
                break

        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def _expand_condition(self, cond_feat: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Broadcast conditioning features to the branch leading shape.

        Args:
            cond_feat: Conditioning features of shape ``(..., cond_dim)``.
            target_shape: Desired leading shape before ``cond_dim``.

        Returns:
            Conditioning features of shape ``(*target_shape, cond_dim)``.
        """
        cond_leading = cond_feat.shape[:-1]
        if cond_leading == target_shape:
            return cond_feat

        if len(cond_leading) == 0:
            view_shape = (1,) * len(target_shape) + cond_feat.shape
            return cond_feat.reshape(view_shape).expand(*target_shape, cond_feat.shape[-1])

        if (
            len(cond_leading) <= len(target_shape)
            and cond_leading == target_shape[-len(cond_leading) :]
        ):
            view_shape = (1,) * (len(target_shape) - len(cond_leading)) + cond_feat.shape
            return cond_feat.reshape(view_shape).expand(*target_shape, cond_feat.shape[-1])

        raise ValueError(
            f"Condition shape {tuple(cond_feat.shape)} cannot broadcast to branch shape "
            f"{tuple(target_shape)} + (cond_dim,)"
        )

    def forward(
        self,
        branches: list[torch.Tensor],
        cond_feat: torch.Tensor,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse active branch tensors.

        Args:
            branches: Active branch tensors with matching leading dimensions.
                Each tensor has shape ``(..., branch_dim_i)``.
            cond_feat: Conditioning features of shape ``(..., cond_dim)`` or
                any trailing-leading shape broadcastable to the branches, such
                as ``(nE, cond_dim)`` for branch tensors shaped ``(B, nE, D)``.
            return_gates: If ``True``, return the normalized gate weights along
                with the fused tensor.

        Returns:
            Fused tensor of shape ``(..., fused_dim)``. If ``return_gates`` is
            ``True``, returns ``(fused, gates)`` where ``gates`` has shape
            ``(..., n_branches)``.
        """
        if len(branches) != self.n_branches:
            raise ValueError(f"Expected {self.n_branches} branches, got {len(branches)}")

        branch_shape = branches[0].shape[:-1]
        for branch in branches[1:]:
            if branch.shape[:-1] != branch_shape:
                raise ValueError("All branch tensors must have matching leading dimensions")

        cond = self._expand_condition(cond_feat, branch_shape)
        proj_branches = [proj(branch) for proj, branch in zip(self.proj, branches)]

        summaries: list[torch.Tensor] = []
        for branch in proj_branches:
            summaries.append(branch)
            summaries.append(torch.abs(branch))

        gate_in = torch.cat(summaries + [cond], dim=-1)
        gate_logits = self.gate_mlp(gate_in)

        if self.use_softmax:
            gates = torch.softmax(gate_logits, dim=-1)
        else:
            gates = torch.sigmoid(gate_logits)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        fused = torch.zeros_like(proj_branches[0])
        for branch_idx, branch in enumerate(proj_branches):
            fused = fused + gates[..., branch_idx].unsqueeze(-1) * branch

        fused = self.out_mlp(fused)

        if return_gates:
            return fused, gates
        return fused
