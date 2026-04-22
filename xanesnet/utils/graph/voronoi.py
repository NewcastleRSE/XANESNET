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
from pymatgen.core import Molecule, Structure
from scipy.spatial import QhullError, Voronoi

from .symmetrize import symmetrize_directed_edges, truncate_per_source


def _polygon_area_3d(verts: np.ndarray) -> float:
    """
    Area of a planar polygon in 3D with ordered vertices ``verts`` of shape
    ``[k, 3]``. Voronoi facets are planar, and ``scipy.spatial.Voronoi``
    returns ridge vertices in a consistent winding order.
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
    """
    Replicate atoms in a supercell around the central cell so that Voronoi
    cells of central-cell atoms are bounded by images rather than the
    supercell boundary.

    Returns ``(points, orig_idx, is_center)`` where
        points     [M, 3]   Cartesian coordinates of all replicated atoms
        orig_idx   [M]      atom index in the original unit cell
        is_center  [M] bool True for atoms in the central image (0, 0, 0)
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
    # neighbours up to ``cutoff`` are inside the supercell.
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
    """
    Run scipy Voronoi on ``points`` and return directed edges between pairs
    that share a (finite) ridge, with at least one endpoint in the central
    image. For every such ridge, both directions (i -> j and j -> i) are
    emitted when both endpoints are central; when only one endpoint is
    central, only that direction is emitted (the reverse comes naturally
    from an equivalent ridge on the other side of the supercell).
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
    """
    Resolve ``min_facet_area`` into an absolute threshold in Å².

    ``None`` -> no filtering (returned as 0.0).
    ``float`` -> absolute threshold.
    ``str`` ending with '%' -> fraction of the maximum facet area in ``areas``.
    Config validity is assumed.
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
    """
    Voronoi-tessellation edge construction.

    Two atoms are connected iff their Voronoi cells share a facet. For
    periodic structures, the tessellation is computed on a supercell so that
    facets with periodic images are resolved correctly; ``edge_weight`` is
    always the Cartesian distance of the edge (which is the correct minimum-
    image distance for periodic graphs, including edges that cross unit cell
    boundaries).

    ``cutoff`` bounds the maximum edge length; facets between atoms farther
    apart than ``cutoff`` are dropped. ``max_num_neighbors`` keeps the
    shortest edges per source node. After truncation, edges are symmetrised
    so the returned graph is guaranteed to be bidirectional.

    ``min_facet_area`` optionally drops edges whose Voronoi facet area is
    below a threshold. It may be a ``float`` (absolute threshold in Å²) or
    a string like ``"1.0%"`` (fraction of the largest facet in this
    structure). ``None`` disables the filter.

    Returns ``(edge_index, edge_weight, edge_vec, edge_attr)`` where
    ``edge_attr`` holds the facet area per edge.
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

    # Optional facet-area filter (absolute Å² or "x%" of max facet area).
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
