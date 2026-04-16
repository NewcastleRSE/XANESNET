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
import torch.nn as nn


class BatchedRadiusGraphBuilder(nn.Module):
    """
    Vectorized padded-batch radius graph construction.

    Returns flattened node indices for a [B, N, ...] tensor layout,
    where flat index = b * N + i.
    """

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, pos: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = pos.device
        dtype = pos.dtype
        bsz, n_atoms, _ = pos.shape

        diff = pos[:, :, None, :] - pos[:, None, :, :]  # [B, N, N, 3]
        dist = torch.linalg.norm(diff, dim=-1)  # [B, N, N]

        valid = mask[:, :, None] & mask[:, None, :]
        edge_mask = valid & (dist <= self.cutoff) & (dist > 1e-8)

        b, src, dst = torch.where(edge_mask)

        if b.numel() == 0:
            edge_src = torch.zeros(0, dtype=torch.long, device=device)
            edge_dst = torch.zeros(0, dtype=torch.long, device=device)
            edge_vec = torch.zeros(0, 3, dtype=dtype, device=device)
        else:
            edge_src = b * n_atoms + src
            edge_dst = b * n_atoms + dst
            edge_vec = pos[b, dst] - pos[b, src]

        return edge_src, edge_dst, edge_vec
