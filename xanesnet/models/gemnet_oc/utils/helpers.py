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

from __future__ import annotations

import torch


def inner_product_clamped(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Clamped inner product of (unit) vectors for numerically-stable acos.
    """
    return (x * y).sum(dim=-1).clamp(min=-1, max=1)


def get_angle(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    Angle between two unit vectors via atan2(|v1 x v2|, v1·v2).
    """
    x = torch.sum(vec1 * vec2, dim=-1)
    y = torch.cross(vec1, vec2, dim=-1).norm(dim=-1)
    y = y.clamp(min=1e-9)
    return torch.atan2(y, x)


def get_inner_idx(idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Return a running 0-based enumeration of duplicates within each group.

    Example: ``idx = [0,0,0,1,2,2]`` -> ``[0,1,2,0,0,1]`` (sorted input).
    For unsorted input, each element receives an enumeration index that is
    unique within its group but in original (unsorted) order; duplicates are
    enumerated by order-of-appearance. This is the behaviour that all XANESNET
    consumers (``target_neighbor_idx`` for edge-scatter, ``out_agg`` for
    ragged triplet aggregation, etc.) actually rely on and matches the
    reference fairchem implementation when the input is sorted.

    The fairchem reference uses ``segment_coo`` which REQUIRES sorted input
    and silently produces garbage otherwise. ``build_edges`` emits edges
    sorted by source (not target), so directly calling the old version on
    ``edge_index[1]`` was incorrect. This version works for any ordering.
    """
    n = idx.size(0)
    if n == 0:
        return idx.new_zeros(0)
    sort_idx = torch.argsort(idx, stable=True)
    counts = torch.bincount(idx, minlength=dim_size)
    group_starts = torch.cumsum(counts, dim=0) - counts
    pos_sorted = torch.arange(n, device=idx.device, dtype=idx.dtype)
    inner_sorted = pos_sorted - group_starts[idx[sort_idx]]
    result = torch.empty_like(idx)
    result[sort_idx] = inner_sorted
    return result
