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

import numpy as np
import torch


def truncate_per_source(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_attr: torch.Tensor | None,
    max_num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Keep at most ``max_num_neighbors`` outgoing edges per source node, choosing
    the shortest ones (by edge_weight). Preserves input ordering among kept
    edges is not required; returned tensors are filtered subsets of inputs.
    """
    if max_num_neighbors is None or max_num_neighbors <= 0:
        return edge_index, edge_weight, edge_vec, edge_attr

    src = edge_index[0]
    e = src.shape[0]
    if e == 0:
        return edge_index, edge_weight, edge_vec, edge_attr

    order = torch.argsort(edge_weight, stable=True).tolist()
    keep = torch.zeros(e, dtype=torch.bool)
    counts: dict[int, int] = {}
    for i in order:
        s = int(src[i].item())
        c = counts.get(s, 0)
        if c < max_num_neighbors:
            keep[i] = True
            counts[s] = c + 1

    edge_index = edge_index[:, keep]
    edge_weight = edge_weight[keep]
    edge_vec = edge_vec[keep]
    edge_attr = edge_attr[keep] if edge_attr is not None else None
    return edge_index, edge_weight, edge_vec, edge_attr


def symmetrize_directed_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_attr: torch.Tensor | None,
    round_decimals: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Keep only directed edges (i -> j, vec) whose reverse counterpart
    (j -> i, -vec) is also present. ``edge_vec`` is used to disambiguate between
    different periodic images of the same atom pair.

    Coordinates are rounded to ``round_decimals`` decimal places to tolerate
    small numerical differences between directions produced by independent
    neighbour searches.
    """
    e = edge_index.shape[1]
    if e == 0:
        return edge_index, edge_weight, edge_vec, edge_attr

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    v = edge_vec.detach().cpu().numpy()
    v_r = np.round(v, round_decimals)
    # Normalise -0.0 -> 0.0 so forward/reverse keys compare equal.
    v_r = v_r + 0.0

    fwd_keys = {(s, d, float(v_r[i, 0]), float(v_r[i, 1]), float(v_r[i, 2])) for i, (s, d) in enumerate(zip(src, dst))}

    mask_list = []
    for i, (s, d) in enumerate(zip(src, dst)):
        rev = (
            d,
            s,
            float(-v_r[i, 0] + 0.0),
            float(-v_r[i, 1] + 0.0),
            float(-v_r[i, 2] + 0.0),
        )
        mask_list.append(rev in fwd_keys)
    mask = torch.tensor(mask_list, dtype=torch.bool)

    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]
    edge_vec = edge_vec[mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None
    return edge_index, edge_weight, edge_vec, edge_attr
