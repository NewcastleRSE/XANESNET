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

"""Unified edge-construction dispatcher for XANESNET graph inputs."""

import torch
from pymatgen.core import Molecule, Structure

from .radius import build_edges_cov_radius, build_edges_radius
from .voronoi import build_edges_voronoi

GRAPH_METHODS = ("radius", "voronoi", "cov_radius")


def build_edges(
    pmg_obj: Structure | Molecule,
    cutoff: float,
    max_num_neighbors: int,
    compute_vectors: bool = True,
    method: str = "radius",
    min_facet_area: float | str | None = None,
    cov_radii_scale: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Dispatch edge construction to the requested graph-building method.

    Supports periodic ``Structure`` and non-periodic ``Molecule`` objects
    uniformly.

    Args:
        pmg_obj: The pymatgen object whose atoms are graph nodes.
        cutoff: Maximum edge length in **angstroms**. For ``"voronoi"`` this
            bounds the edge length after tessellation. For ``"cov_radius"``
            this serves as a hard upper bound (the effective cutoff is the
            smaller of ``cutoff`` and ``cov_radii_scale * (r_cov_src + r_cov_dst)``).
        max_num_neighbors: Maximum outgoing edges kept per source node
            (shortest first). The returned graph is guaranteed to be
            bidirectional regardless of this truncation.
        compute_vectors: If ``False``, ``edge_vec`` is returned as ``None``
            to save memory in downstream consumers that do not need it.
            Geometry-aware methods still compute vectors internally (they are
            needed for symmetrisation).
        method: One of ``"radius"`` (default), ``"voronoi"``, or ``"cov_radius"``.
        min_facet_area: Only used by ``"voronoi"``. See ``build_edges_voronoi``.
        cov_radii_scale: Only used by ``"cov_radius"``. See ``build_edges_cov_radius``.

    Returns:
        A 4-tuple ``(edge_index, edge_weight, edge_vec, edge_attr)``:

        - ``edge_index``: ``(2, E)`` int64 -- source/destination node indices.
        - ``edge_weight``: ``(E,)`` float32 -- edge length in **angstroms**.
        - ``edge_vec``: ``(E, 3)`` float32 or ``None``.
        - ``edge_attr``: ``(E,)`` float32 or ``None`` -- extra per-edge scalar
          (facet area for ``"voronoi"``, ``None`` otherwise).

    Raises:
        ValueError: If ``method`` is not one of ``GRAPH_METHODS``.
    """
    method = method.lower()
    if method == "radius":
        edge_index, edge_weight, edge_vec, edge_attr = build_edges_radius(pmg_obj, cutoff, max_num_neighbors)
    elif method == "voronoi":
        edge_index, edge_weight, edge_vec, edge_attr = build_edges_voronoi(
            pmg_obj, cutoff, max_num_neighbors, min_facet_area=min_facet_area
        )
    elif method == "cov_radius":
        edge_index, edge_weight, edge_vec, edge_attr = build_edges_cov_radius(
            pmg_obj, cutoff, max_num_neighbors, cov_radii_scale=cov_radii_scale
        )
    else:
        raise ValueError(f"Unknown graph method {method!r}; expected one of {GRAPH_METHODS}")

    if not compute_vectors:
        edge_vec = None
    return edge_index, edge_weight, edge_vec, edge_attr
