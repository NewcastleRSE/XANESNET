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


def build_absorber_relative_geometry(
    pos: torch.Tensor,
    mask: torch.Tensor,
    absorber_index: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Build absorber-relative geometry once and reuse it everywhere.

    Args:
        pos:            [B, N, 3] atom positions
        mask:           [B, N] valid-atom mask
        absorber_index: index of the absorber atom (default 0)

    Returns dict with:
        rel         [B, N, 3]  position relative to absorber
        r           [B, N]     absorber-neighbour distance
        u           [B, N, 3]  absorber-neighbour unit vector
        valid_neigh [B, N]     valid neighbours excluding absorber
    """
    abs_pos = pos[:, absorber_index, :].unsqueeze(1)  # [B, 1, 3]
    rel = pos - abs_pos  # [B, N, 3]
    r = torch.linalg.norm(rel, dim=-1)  # [B, N]
    u = rel / r.unsqueeze(-1).clamp_min(1e-8)  # [B, N, 3]

    valid_neigh = mask.clone()
    valid_neigh[:, absorber_index] = False

    return {
        "rel": rel,
        "r": r,
        "u": u,
        "valid_neigh": valid_neigh,
    }
