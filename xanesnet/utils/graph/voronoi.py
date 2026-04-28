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

"""Voronoi-tessellation edge construction for XANESNET graph inputs."""

import numpy as np
import torch
from pymatgen.core import Molecule, Structure
from scipy.spatial import QhullError, Voronoi

from .symmetrize import symmetrize_directed_edges, truncate_per_source


def _polygon_area_3d(verts: np.ndarray) -> float:
    """Compute the area of a planar polygon in 3D from its ordered vertices.

    Args:
        verts: ``(K, 3)`` float -- ordered polygon vertices. Must contain at
            least 3 points. Voronoi facets are planar, and
            ``scipy.spatial.Voronoi`` returns ridge vertices in a consistent
            winding order.

    Returns:
        Area of the polygon in the same squared units as ``verts``.
    """
    n = verts.shape[0]
    if n < 3:
        return 0.0
    cross_sum = np.zeros(3, dtype=np.float64)
    for i in range(n):
        cross_sum += np.cross(verts[i], verts[(i + 1) % n])
    return 0.5 * float(np.linalg.norm(cross_sum))


def _build_periodic_supercell(
    structure: Structure,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate a periodic structure into a supercell for Voronoi tessellation.

    The supercell is sized so that the Voronoi cells of central-image atoms are
    bounded by replica atoms rather than the supercell boundary, up to
    ``cutoff``. The number of replicas per direction is determined from the
    perpendicular lattice-plane spacings (not the lattice vector lengths), so
    oblique cells are handled correctly.

    Args:
        structure: Periodic pymatgen ``Structure``.
        cutoff: Maximum edge length considered in **angstroms**. Controls the
            minimum supercell extent.

    Returns:
        A 3-tuple ``(points, orig_idx, is_center)``:

                - ``points``: ``(M, 3)`` float64 -- Cartesian coordinates of all
          replicated atoms.
                - ``orig_idx``: ``(M,)`` int64 -- atom index in the original unit
          cell.
                - ``is_center``: ``(M,)`` bool -- ``True`` for atoms in the central
          image ``(0, 0, 0)``.
    """
    lat = np.array(structure.lattice.matrix, dtype=np.float64)
    # Use *perpendicular* spacings between opposite lattice planes (not |a|,|b|,|c|)
    # so that oblique cells get enough replicas for central-cell Voronoi cells to
    # be fully bounded by image atoms up to ``cutoff``.
    volume = float(abs(np.linalg.det(lat)))
    perp = np.array(
        [
            volume / float(np.linalg.norm(np.cross(lat[1], lat[2]))),
            volume / float(np.linalg.norm(np.cross(lat[2], lat[0]))),
            volume / float(np.linalg.norm(np.cross(lat[0], lat[1]))),
        ],
        dtype=np.float64,
    )
    # Number of images each direction: enough that any central atom's Voronoi
    # neighbors up to ``cutoff`` are inside the supercell.
    n_reps = np.maximum(1, np.ceil(cutoff / perp).astype(int))

    coords = np.array(structure.cart_coords, dtype=np.float64)
    n_atoms = coords.shape[0]

    pts_list: list[np.ndarray] = []
    idx_list: list[np.ndarray] = []
    center_list: list[np.ndarray] = []
    for a in range(-int(n_reps[0]), int(n_reps[0]) + 1):
        for b in range(-int(n_reps[1]), int(n_reps[1]) + 1):
            for c in range(-int(n_reps[2]), int(n_reps[2]) + 1):
                shift = a * lat[0] + b * lat[1] + c * lat[2]
                pts_list.append(coords + shift)
                idx_list.append(np.arange(n_atoms, dtype=np.int64))
                is_center = (a == 0) and (b == 0) and (c == 0)
                center_list.append(np.full(n_atoms, is_center, dtype=bool))

    return (
        np.concatenate(pts_list, axis=0),
        np.concatenate(idx_list, axis=0),
        np.concatenate(center_list, axis=0),
    )


def _voronoi_edges(
    points: np.ndarray,
    is_center: np.ndarray,
    orig_idx: np.ndarray,
    cutoff: float,
) -> tuple[list[int], list[int], list[float], list[np.ndarray], list[float]]:
    """Extract directed edges from a Voronoi tessellation.

    Runs ``scipy.spatial.Voronoi`` on ``points`` and returns directed edges
    between pairs that share a finite ridge, with at least one endpoint in
    the central image. For ridges with both endpoints in the central image,
    both directions are emitted; for ridges with only one central endpoint,
    only that outgoing direction is emitted (the reverse arrives from an
    equivalent ridge on the other side of the supercell).

    Ridges with fewer than 3 vertices or any infinite vertex (index ``-1``)
    are skipped.

    Args:
                points: ``(M, 3)`` float64 -- Cartesian coordinates of all atoms
          (including supercell replicas).
                is_center: ``(M,)`` bool -- ``True`` for central-image atoms.
                orig_idx: ``(M,)`` int64 -- original unit-cell atom index per point.
        cutoff: Maximum edge length in **angstroms**; longer ridges are
            dropped.

    Returns:
        A 5-tuple ``(src, dst, dist, vec, area)`` of equal-length lists:

        - ``src``: source atom indices (original unit-cell).
        - ``dst``: destination atom indices (original unit-cell).
        - ``dist``: edge lengths in **angstroms**.
        - ``vec``: displacement vectors ``(3,)`` float32 each.
        - ``area``: Voronoi facet areas in **angstroms squared**.
    """
    if points.shape[0] < 4:
        return [], [], [], [], []
    try:
        vor = Voronoi(points)
    except QhullError:
        return [], [], [], [], []

    src: list[int] = []
    dst: list[int] = []
    dist: list[float] = []
    vec: list[np.ndarray] = []
    area: list[float] = []

    ridge_points = vor.ridge_points
    ridge_vertices = vor.ridge_vertices
    vor_vertices = vor.vertices

    for rp, rv in zip(ridge_points, ridge_vertices):
        if len(rv) < 3 or -1 in rv:
            continue
        p0, p1 = int(rp[0]), int(rp[1])
        c0 = bool(is_center[p0])
        c1 = bool(is_center[p1])
        if not (c0 or c1):
            continue

        displacement = points[p1] - points[p0]
        d = float(np.linalg.norm(displacement))
        if d > cutoff or d < 1e-8:
            continue

        a = _polygon_area_3d(vor_vertices[rv])

        if c0:
            src.append(int(orig_idx[p0]))
            dst.append(int(orig_idx[p1]))
            dist.append(d)
            vec.append(displacement.astype(np.float32))
            area.append(a)
        if c1:
            src.append(int(orig_idx[p1]))
            dst.append(int(orig_idx[p0]))
            dist.append(d)
            vec.append((-displacement).astype(np.float32))
            area.append(a)

    return src, dst, dist, vec, area


def _resolve_min_facet_area(
    min_facet_area: float | str | None,
    areas: np.ndarray,
) -> float:
    """Resolve ``min_facet_area`` to an absolute area threshold.

    Args:
        min_facet_area: Filtering specification. ``None`` disables filtering
            (returns ``0.0``). A ``float`` is used as an absolute threshold
            in **angstroms squared**. A ``str`` ending with ``'%'`` is
            interpreted as a fraction of the maximum facet area in ``areas``.
        areas: ``(E,)`` float -- facet areas used when resolving percentage
            thresholds.

    Returns:
        Absolute area threshold in **angstroms squared**. Returns ``0.0``
        when filtering is disabled.
    """
    if min_facet_area is None:
        return 0.0
    if isinstance(min_facet_area, str):
        pct = float(min_facet_area.rstrip("%").strip()) / 100.0
        amax = float(areas.max()) if areas.size > 0 else 0.0
        return pct * amax
    return float(min_facet_area)


def build_edges_voronoi(
    pmg_obj: Structure | Molecule,
    cutoff: float,
    max_num_neighbors: int,
    min_facet_area: float | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voronoi-tessellation edge construction.

    Two atoms are connected iff their Voronoi cells share a facet. For
    periodic structures, the tessellation is computed on a supercell so that
    facets with periodic images are resolved correctly. ``edge_weight`` is
    always the Cartesian distance of the edge, which equals the correct
    minimum-image distance for periodic graphs.

    After optional facet-area filtering, edges are truncated to
    ``max_num_neighbors`` per source node and symmetrised to guarantee a
    bidirectional graph.

    Args:
        pmg_obj: Periodic ``Structure`` or non-periodic ``Molecule``.
        cutoff: Maximum edge length in **angstroms**. Facets between atoms
            farther apart are dropped.
        max_num_neighbors: Maximum outgoing edges retained per source node
            (shortest first).
        min_facet_area: Optional lower bound on the Voronoi facet area.
            ``None`` disables the filter. A ``float`` is an absolute threshold
            in **angstroms squared**. A ``str`` like ``"1.0%"`` is a fraction
            of the largest facet area in the structure.

    Returns:
        A 4-tuple ``(edge_index, edge_weight, edge_vec, edge_attr)``:

        - ``edge_index``: ``(2, E)`` int64.
                - ``edge_weight``: ``(E,)`` float32 -- edge lengths in **angstroms**.
                - ``edge_vec``: ``(E, 3)`` float32 -- displacement vectors.
                - ``edge_attr``: ``(E,)`` float32 -- Voronoi facet areas in
          **angstroms squared**.
    """
    if isinstance(pmg_obj, Structure):
        points, orig_idx, is_center = _build_periodic_supercell(pmg_obj, cutoff)
    else:
        coords = np.array(pmg_obj.cart_coords, dtype=np.float64)
        n = coords.shape[0]
        points = coords
        orig_idx = np.arange(n, dtype=np.int64)
        is_center = np.ones(n, dtype=bool)

    src_l, dst_l, dist_l, vec_l, area_l = _voronoi_edges(points, is_center, orig_idx, cutoff)

    if len(src_l) == 0:
        return (
            torch.zeros(2, 0, dtype=torch.int64),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, 3, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )

    edge_index = torch.tensor([src_l, dst_l], dtype=torch.int64)
    edge_weight = torch.tensor(dist_l, dtype=torch.float32)
    edge_vec = torch.tensor(np.stack(vec_l, axis=0), dtype=torch.float32)
    edge_attr = torch.tensor(area_l, dtype=torch.float32)

    # Optional facet-area filter (absolute angstroms^2 or "x%" of max facet area).
    area_threshold = _resolve_min_facet_area(min_facet_area, edge_attr.numpy())
    if area_threshold > 0.0:
        keep = edge_attr >= area_threshold
        edge_index = edge_index[:, keep]
        edge_weight = edge_weight[keep]
        edge_vec = edge_vec[keep]
        edge_attr = edge_attr[keep]

    edge_index, edge_weight, edge_vec, edge_attr = truncate_per_source(
        edge_index, edge_weight, edge_vec, edge_attr, max_num_neighbors
    )
    edge_index, edge_weight, edge_vec, edge_attr = symmetrize_directed_edges(
        edge_index, edge_weight, edge_vec, edge_attr
    )
    # edge_attr is never dropped by the helpers when a Tensor is passed in.
    assert edge_attr is not None
    return edge_index, edge_weight, edge_vec, edge_attr
