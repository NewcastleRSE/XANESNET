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

"""Edge-list post-processing: per-source truncation and bidirectionality enforcement."""

import numpy as np
import torch


def _quantized_edge_keys(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    round_decimals: int,
) -> list[tuple[int, int, float, float, float]]:
    """Return rounded edge keys used for reverse-edge matching.

    Args:
        edge_index: ``(2, E)`` int64 source/destination node indices.
        edge_vec: ``(E, 3)`` float displacement vectors.
        round_decimals: Decimal places used before matching.

    Returns:
        List of keys ``(src, dst, vx, vy, vz)``.
    """
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    v = edge_vec.detach().cpu().numpy()
    v_r = np.round(v, round_decimals)
    v_r = v_r + 0.0
    return [
        (src_i, dst_i, float(vec_i[0]), float(vec_i[1]), float(vec_i[2]))
        for src_i, dst_i, vec_i in zip(src, dst, v_r, strict=True)
    ]


def _reverse_key(key: tuple[int, int, float, float, float]) -> tuple[int, int, float, float, float]:
    """Return the reverse-edge key corresponding to ``key``.

    Args:
        key: Edge key ``(src, dst, vx, vy, vz)``.

    Returns:
        Reverse-edge key ``(dst, src, -vx, -vy, -vz)``.
    """
    src, dst, vx, vy, vz = key
    return dst, src, float(-vx + 0.0), float(-vy + 0.0), float(-vz + 0.0)


def truncate_per_source(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_attr: torch.Tensor | None,
    max_num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Keep at most ``max_num_neighbors`` outgoing edges per source node.

    Edges are selected greedily by ascending ``edge_weight`` (shortest first).
    The relative ordering of the kept edges is not guaranteed to be preserved.

    Args:
        edge_index: ``(2, E)`` int64 -- source/destination node indices.
        edge_weight: ``(E,)`` float -- edge lengths.
        edge_vec: ``(E, 3)`` float -- displacement vectors.
        edge_attr: ``(E,)`` float or ``None`` -- optional per-edge scalar.
        max_num_neighbors: Maximum number of outgoing edges to retain per
            source node. Values ``<= 0`` or ``None`` disable truncation.

    Returns:
        Filtered ``(edge_index, edge_weight, edge_vec, edge_attr)`` with the
        same tensor types as the inputs.
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
    """Retain only edges whose reverse counterpart is also present.

    An edge ``(i -> j, vec)`` is kept iff ``(j -> i, -vec)`` also exists in
    the edge list. ``edge_vec`` is used to disambiguate between different
    periodic images of the same atom pair.

    Matching is performed with multiplicity: if one direction appears more
    times than its reverse after rounding, only the matched pairs are kept.

    Coordinates are rounded to ``round_decimals`` decimal places to tolerate
    small numerical differences between forward and reverse vectors produced
    by independent neighbor searches.

    Args:
        edge_index: ``(2, E)`` int64 -- source/destination node indices.
        edge_weight: ``(E,)`` float -- edge lengths.
        edge_vec: ``(E, 3)`` float -- displacement vectors.
        edge_attr: ``(E,)`` float or ``None`` -- optional per-edge scalar.
        round_decimals: Number of decimal places used when comparing
            displacement vectors for forward/reverse matching.

    Returns:
        Filtered ``(edge_index, edge_weight, edge_vec, edge_attr)`` containing
        only bidirectional edges.
    """
    e = edge_index.shape[1]
    if e == 0:
        return edge_index, edge_weight, edge_vec, edge_attr

    keys = _quantized_edge_keys(edge_index, edge_vec, round_decimals)
    buckets: dict[tuple[int, int, float, float, float], list[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault(key, []).append(i)

    mask = torch.zeros(e, dtype=torch.bool)
    processed: set[tuple[int, int, float, float, float]] = set()
    for key, idxs in buckets.items():
        if key in processed:
            continue

        rev_key = _reverse_key(key)
        rev_idxs = buckets.get(rev_key)
        if rev_idxs is None:
            processed.add(key)
            continue

        if key == rev_key:
            mask[idxs] = True
            processed.add(key)
            continue

        matched = min(len(idxs), len(rev_idxs))
        if matched > 0:
            mask[idxs[:matched]] = True
            mask[rev_idxs[:matched]] = True

        processed.add(key)
        processed.add(rev_key)

    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]
    edge_vec = edge_vec[mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None
    return edge_index, edge_weight, edge_vec, edge_attr
