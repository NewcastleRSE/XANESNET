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
from torch_geometric.typing import SparseTensor


def compute_triplets_and_angles(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    num_nodes: int,
    is_periodic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute triplets (k->j->i) and the angle at node j for each triplet.

    Returns (angle, idx_kj, idx_ji).
    """
    row, col = edge_index

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col,
        col=row,
        value=value,
        sparse_sizes=(num_nodes, num_nodes),
    )
    adj_t_row = adj_t.index_select(0, row)  # type: ignore[attr-defined]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets
    idx_i = col.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    # Edge indices (k->j, j->i) for triplets
    idx_kj_raw = adj_t_row.storage.value()
    idx_ji_raw = adj_t_row.storage.row()

    # Remove degenerate triplets.
    if is_periodic:
        # For periodic structures, idx_i == idx_k does NOT imply a degenerate
        # bounce-back: k and i may be different periodic images of the same atom.
        # Reject triplets where the same edge serves as both legs, and also
        # same-image bounce-back (k->j and j->k via the reverse edge of the same
        # bond), which would give vec_ji == vec_jk and a spurious 0-degree angle.
        # Same-image bounce-back <=> edge_vec[idx_ji] == -edge_vec[idx_kj].
        bounce = (edge_vec[idx_ji_raw] + edge_vec[idx_kj_raw]).norm(dim=-1) < 1e-6
        mask = (idx_kj_raw != idx_ji_raw) & ~bounce
    else:
        # For molecules, each atom pair has exactly one edge per direction,
        # so idx_i == idx_k correctly identifies bounce-back triplets.
        mask = idx_i != idx_k

    idx_kj = idx_kj_raw[mask]
    idx_ji = idx_ji_raw[mask]

    # Compute the angle at node j (the intermediate node in the triplet k->j->i).
    # vec_ji: j→i displacement, vec_jk: j→k displacement (negated k→j edge).
    vec_ji = edge_vec[idx_ji]
    vec_jk = -edge_vec[idx_kj]

    a = (vec_ji * vec_jk).sum(dim=-1)
    b = torch.cross(vec_ji, vec_jk, dim=1).norm(dim=-1)
    angle = torch.atan2(b, a)

    return angle, idx_kj, idx_ji
