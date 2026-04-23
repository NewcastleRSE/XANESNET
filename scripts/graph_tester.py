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

"""
Graph construction visual tester.

Standalone diagnostic script to visually confirm that the graph-building logic
used by GeometryGraphDataset and E3EEDataset behaves correctly for both periodic
Structures and non-periodic Molecules. Useful when picking sensible values for
``cutoff`` and ``max_num_neighbors``.

Input is a ``pmgjson`` datasource directory (same layout as used by the rest
of XANESNET) - each ``.json`` holds a single pymatgen Structure or Molecule.
A sample is selected by index or by file stem.

Produces three 3D views of the same structure - edges only, geometrygraph
triplets only, e3ee absorber paths only - plus histograms of edge weights,
per-atom out-degree, and angle distributions.

Run:
    python scripts/graph_tester.py --json-dir data/fe/json_train --index 0 \
        --cutoff 6.0 --max-neighbors 32 --graph-method voronoi --show-voronoi

This script does NOT import any model code; it only exercises the shared
``xanesnet.utils.graph`` helpers, matching what the datasets do at prepare().
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from pymatgen.core import Element, Molecule, Structure

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xanesnet.datasources.pmgjson import PMGJSONSource
from xanesnet.utils.graph import (
    GRAPH_METHODS,
    build_absorber_paths,
    build_edges,
    compute_triplets_and_angles,
)


def load_sample(json_dir: Path, index: int | None, file_stem: str | None) -> Structure | Molecule:
    ds = PMGJSONSource(datasource_type="pmgjson", json_path=str(json_dir))
    if file_stem is not None:
        if file_stem not in ds.file_names:
            raise SystemExit(f"file stem {file_stem!r} not found in {json_dir}")
        target_idx = ds.file_names.index(file_stem)
    else:
        target_idx = 0 if index is None else int(index)
        if not (0 <= target_idx < len(ds)):
            raise SystemExit(f"--index {target_idx} out of range [0, {len(ds)})")
    for i, obj in enumerate(ds):
        if i == target_idx:
            return obj
    raise SystemExit("Failed to iterate datasource")


def element_colors(atomic_numbers: np.ndarray) -> list[tuple[float, float, float]]:
    colors: list[tuple[float, float, float]] = []
    for z in atomic_numbers.tolist():
        try:
            rgb = Element.from_Z(int(z)).color
            colors.append(tuple(float(c) for c in rgb))  # type: ignore[arg-type]
        except Exception:
            colors.append((0.5, 0.5, 0.5))
    return colors


def _cell_corner_coords(structure: Structure) -> np.ndarray:
    lattice = np.array(structure.lattice.matrix, dtype=np.float64)
    return np.array(list(itertools.product([0.0, 1.0], repeat=3))) @ lattice


def draw_unit_cell(ax, structure: Structure) -> None:
    corners_frac = np.array(list(itertools.product([0.0, 1.0], repeat=3)))
    corners = _cell_corner_coords(structure)
    edges = []
    for i, j in itertools.combinations(range(8), 2):
        if np.sum(np.abs(corners_frac[i] - corners_frac[j])) == 1.0:
            edges.append([corners[i], corners[j]])
    ax.add_collection3d(Line3DCollection(edges, colors="black", linewidths=0.6, alpha=0.35))


def plot_atoms(ax, coords, atomic_numbers, absorber_idx, label=True):
    colors = element_colors(atomic_numbers)
    sizes = np.array([max(60.0, 30.0 + float(z) * 2.0) for z in atomic_numbers])
    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=sizes, edgecolors="k", linewidths=0.4, depthshade=True
    )
    ax.scatter(
        coords[absorber_idx, 0],
        coords[absorber_idx, 1],
        coords[absorber_idx, 2],
        s=220,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
    )
    if label:
        for i, (x, y, z_) in enumerate(coords):
            sym = Element.from_Z(int(atomic_numbers[i])).symbol
            ax.text(x, y, z_, f" {sym}{i}", fontsize=6, color="black")


def plot_edges(ax, coords, edge_src, edge_dst, edge_vec, is_periodic):
    """
    Draw intra-cell edges boldly; PBC-crossing edges thinner and more faded so focus stays on the primary graph.
    """
    segments = []
    segment_colors = []
    segment_widths = []
    n_crossing = 0
    for s, d, vec in zip(edge_src.tolist(), edge_dst.tolist(), edge_vec):
        start = coords[s]
        end = start + np.asarray(vec, dtype=np.float64)
        segments.append(np.stack([start, end], axis=0))
        if is_periodic:
            delta = np.linalg.norm(end - coords[d])
            if delta > 1e-6:
                segment_colors.append((0.85, 0.30, 0.10, 0.55))
                segment_widths.append(0.5)
                n_crossing += 1
            else:
                segment_colors.append((0.15, 0.45, 0.75, 0.8))
                segment_widths.append(1.4)
        else:
            segment_colors.append((0.15, 0.45, 0.75, 0.75))
            segment_widths.append(1.2)
    ax.add_collection3d(Line3DCollection(segments, colors=segment_colors, linewidths=segment_widths))
    return n_crossing


def plot_triplets(ax, coords, edge_src, edge_dst, edge_vec, idx_kj, idx_ji, max_draw):
    n = idx_kj.shape[0]
    if n == 0:
        return
    take = np.random.default_rng(0).choice(n, size=min(max_draw, n), replace=False)
    tris = []
    for t in take:
        ekj = int(idx_kj[t])
        eji = int(idx_ji[t])
        j = int(edge_dst[ekj])
        pj = coords[j]
        pk = pj - np.asarray(edge_vec[ekj], dtype=np.float64)
        pi = pj + np.asarray(edge_vec[eji], dtype=np.float64)
        tris.append(np.stack([pk, pj, pi], axis=0))
    poly = Poly3DCollection(
        tris, facecolors=(0.30, 0.75, 0.35, 0.22), edgecolors=(0.10, 0.50, 0.15, 0.7), linewidths=0.7
    )
    ax.add_collection3d(poly)


def absorber_neighbor_coords(pmg_obj, absorber_idx, cutoff):
    abs_coord = np.array(pmg_obj.cart_coords[absorber_idx], dtype=np.float64)
    if isinstance(pmg_obj, Structure):
        neighbors = pmg_obj.get_neighbors(pmg_obj[absorber_idx], r=cutoff)
        if not neighbors:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array([n.coords for n in neighbors], dtype=np.float64)
    all_coords = np.array(pmg_obj.cart_coords, dtype=np.float64)
    dists = np.linalg.norm(all_coords - abs_coord, axis=-1)
    keep = (dists <= cutoff) & (np.arange(len(pmg_obj)) != absorber_idx)
    return all_coords[keep]


def plot_absorber_paths(ax, pmg_obj, absorber_idx, cutoff, max_paths, max_draw):
    abs_coord = np.array(pmg_obj.cart_coords[absorber_idx], dtype=np.float64)
    neigh_coords = absorber_neighbor_coords(pmg_obj, absorber_idx, cutoff)
    n = neigh_coords.shape[0]
    if n < 2:
        return
    ii, jj = np.triu_indices(n, k=1)
    cj, ck = neigh_coords[ii], neigh_coords[jj]
    r0j = np.linalg.norm(cj - abs_coord, axis=-1)
    r0k = np.linalg.norm(ck - abs_coord, axis=-1)
    rjk = np.linalg.norm(ck - cj, axis=-1)
    score = r0j + r0k + 0.5 * rjk
    order = np.argsort(score)[:max_paths]
    draw_order = order[: min(max_draw, order.shape[0])]
    tris = [np.stack([abs_coord, cj[t], ck[t]], axis=0) for t in draw_order]
    poly = Poly3DCollection(
        tris, facecolors=(0.85, 0.25, 0.55, 0.20), edgecolors=(0.70, 0.10, 0.40, 0.75), linewidths=0.7
    )
    ax.add_collection3d(poly)


def equalize_3d_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = float((maxs - mins).max()) * 0.55 + 1e-6
    center = 0.5 * (mins + maxs)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def setup_axis(ax, pmg_obj, coords, atomic_numbers, absorber_idx, vis_points, is_periodic, title, label_atoms):
    if is_periodic:
        draw_unit_cell(ax, pmg_obj)
    plot_atoms(ax, coords, atomic_numbers, absorber_idx, label=label_atoms)
    ax.set_title(title, fontsize=10)
    equalize_3d_axes(ax, vis_points)


def compute_voronoi_facets(pmg_obj, cutoff):
    """
    Return a list of (polygon_verts [k,3], area, pair_distance) for Voronoi
    facets where at least one endpoint is in the central cell (or, for a
    Molecule, any finite facet). Edges whose midpoint-pair distance exceeds
    ``cutoff`` are dropped, matching ``build_edges_voronoi``.

    Independent scipy reimplementation so the script does not reach into
    private helpers of ``xanesnet.utils.graph``.
    """
    from scipy.spatial import (  # local import keeps import cost optional
        QhullError,
        Voronoi,
    )

    if isinstance(pmg_obj, Structure):
        lat = np.array(pmg_obj.lattice.matrix, dtype=np.float64)
        volume = float(abs(np.linalg.det(lat)))
        perp = np.array(
            [
                volume / float(np.linalg.norm(np.cross(lat[1], lat[2]))),
                volume / float(np.linalg.norm(np.cross(lat[2], lat[0]))),
                volume / float(np.linalg.norm(np.cross(lat[0], lat[1]))),
            ],
            dtype=np.float64,
        )
        n_reps = np.maximum(1, np.ceil(cutoff / perp).astype(int))
        base = np.array(pmg_obj.cart_coords, dtype=np.float64)
        pts: list[np.ndarray] = []
        is_center_list: list[np.ndarray] = []
        for a in range(-int(n_reps[0]), int(n_reps[0]) + 1):
            for b in range(-int(n_reps[1]), int(n_reps[1]) + 1):
                for c in range(-int(n_reps[2]), int(n_reps[2]) + 1):
                    shift = a * lat[0] + b * lat[1] + c * lat[2]
                    pts.append(base + shift)
                    is_center_list.append(np.full(base.shape[0], (a == 0 and b == 0 and c == 0), dtype=bool))
        points = np.concatenate(pts, axis=0)
        is_center = np.concatenate(is_center_list, axis=0)
    else:
        points = np.array(pmg_obj.cart_coords, dtype=np.float64)
        is_center = np.ones(points.shape[0], dtype=bool)

    if points.shape[0] < 4:
        return []
    try:
        vor = Voronoi(points)
    except QhullError:
        return []

    facets: list[tuple[np.ndarray, float, float]] = []
    for rp, rv in zip(vor.ridge_points, vor.ridge_vertices):
        if len(rv) < 3 or -1 in rv:
            continue
        p0, p1 = int(rp[0]), int(rp[1])
        if not (bool(is_center[p0]) or bool(is_center[p1])):
            continue
        d = float(np.linalg.norm(points[p1] - points[p0]))
        if d > cutoff or d < 1e-8:
            continue
        verts = vor.vertices[rv]
        # Defensive: sort vertices around the facet centroid in the facet plane
        # so the shoelace cross-sum is correct even for non-convex winding.
        centroid = verts.mean(axis=0)
        rel = verts - centroid
        # Estimate facet normal from first non-degenerate triangle
        normal = np.cross(rel[1], rel[2])
        nrm = float(np.linalg.norm(normal))
        if nrm < 1e-12:
            continue
        normal = normal / nrm
        u = rel[0] / (np.linalg.norm(rel[0]) + 1e-18)
        v = np.cross(normal, u)
        ang = np.arctan2(rel @ v, rel @ u)
        order = np.argsort(ang)
        verts_sorted = verts[order]
        cross_sum = np.zeros(3, dtype=np.float64)
        for k in range(verts_sorted.shape[0]):
            cross_sum += np.cross(verts_sorted[k], verts_sorted[(k + 1) % verts_sorted.shape[0]])
        area = 0.5 * float(np.linalg.norm(cross_sum))
        facets.append((verts_sorted, area, d))
    return facets


def plot_voronoi_facets(ax, facets):
    if not facets:
        return
    polys = [f[0] for f in facets]
    areas = np.array([f[1] for f in facets], dtype=np.float64)
    if areas.size == 0:
        return
    # Color facets by area on a perceptual colormap so small and large facets
    # are both clearly distinguishable against the white background and atoms.
    amax = float(areas.max()) if areas.max() > 0 else 1.0
    amin = float(areas.min())
    norm = (areas - amin) / max(amax - amin, 1e-12)
    cmap = plt.get_cmap("viridis")
    face_colors = [(*cmap(float(n))[:3], 0.30) for n in norm]
    edge_colors = [(*cmap(float(n))[:3], 0.95) for n in norm]
    coll = Poly3DCollection(
        polys,
        facecolors=face_colors,
        edgecolors=edge_colors,
        linewidths=1.0,
    )
    ax.add_collection3d(coll)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json-dir", required=True, type=Path, help="PMGJSON datasource directory (one .json per sample)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--index", type=int, help="Sample index into sorted datasource")
    g.add_argument("--file", type=str, help="Sample file stem (without .json)")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--max-neighbors", type=int, default=32)
    p.add_argument(
        "--graph-method",
        type=str,
        default="radius",
        choices=list(GRAPH_METHODS),
        help="Edge construction method (matches xanesnet.utils.graph.build_edges)",
    )
    p.add_argument(
        "--min-facet-area",
        type=str,
        default=None,
        help="Voronoi only: drop facets below this area. Absolute float in A^2 "
        "(e.g. 0.25) or a percentage of the max facet area (e.g. '1.0%%').",
    )
    p.add_argument(
        "--cov-radii-scale",
        type=float,
        default=1.5,
        help="cov_radius only: scale factor on (r_cov_i + r_cov_j).",
    )
    p.add_argument(
        "--show-voronoi",
        action="store_true",
        help="Overlay Voronoi tessellation facets on the edges panel (any method)",
    )
    p.add_argument("--absorber-idx", type=int, default=0)
    p.add_argument("--max-triplets-drawn", type=int, default=60)
    p.add_argument("--max-paths-drawn", type=int, default=60)
    p.add_argument("--max-paths", type=int, default=128, help="Truncation budget for absorber paths")
    p.add_argument("--no-atom-labels", action="store_true")
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    pmg_obj = load_sample(args.json_dir, args.index, args.file)
    stem = pmg_obj.properties.get("file_name", "<unknown>")
    is_periodic = isinstance(pmg_obj, Structure)
    coords = np.array(pmg_obj.cart_coords, dtype=np.float64)
    atomic_numbers = np.array(pmg_obj.atomic_numbers, dtype=np.int64)
    n_atoms = len(pmg_obj)

    if not (0 <= args.absorber_idx < n_atoms):
        raise SystemExit(f"--absorber-idx {args.absorber_idx} out of range [0, {n_atoms})")

    # min_facet_area is a float-or-string: coerce string-looking floats to float
    min_facet_area = args.min_facet_area
    if min_facet_area is not None and not min_facet_area.endswith("%"):
        min_facet_area = float(min_facet_area)

    edge_index, edge_weight, edge_vec, edge_attr = build_edges(
        pmg_obj,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_neighbors,
        compute_vectors=True,
        method=args.graph_method,
        min_facet_area=min_facet_area,
        cov_radii_scale=args.cov_radii_scale,
    )
    assert edge_vec is not None
    edge_src = edge_index[0].numpy()
    edge_dst = edge_index[1].numpy()
    edge_vec_np = edge_vec.numpy()
    edge_w_np = edge_weight.numpy()

    angle, idx_kj, idx_ji = compute_triplets_and_angles(
        edge_index,
        edge_vec,
        num_nodes=n_atoms,
        is_periodic=is_periodic,
    )
    angle_np = angle.numpy()
    idx_kj_np = idx_kj.numpy()
    idx_ji_np = idx_ji.numpy()

    paths = build_absorber_paths(
        pmg_obj,
        absorber_idx=args.absorber_idx,
        cutoff=args.cutoff,
        max_paths=args.max_paths,
    )

    vis_points = coords.copy()
    if is_periodic:
        vis_points = np.concatenate([vis_points, _cell_corner_coords(pmg_obj)], axis=0)
    edge_ends = coords[edge_src] + edge_vec_np
    if edge_ends.shape[0] > 0:
        vis_points = np.concatenate([vis_points, edge_ends], axis=0)

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[3.0, 1.3],
        hspace=0.35,
        wspace=0.28,
        left=0.05,
        right=0.97,
        top=0.94,
        bottom=0.08,
    )
    ax_edges = fig.add_subplot(gs[0, 0], projection="3d")
    ax_trip = fig.add_subplot(gs[0, 1], projection="3d")
    ax_paths = fig.add_subplot(gs[0, 2], projection="3d")
    ax_hist_w = fig.add_subplot(gs[1, 0])
    ax_hist_deg = fig.add_subplot(gs[1, 1])
    ax_hist_ang = fig.add_subplot(gs[1, 2])

    label_atoms = (not args.no_atom_labels) and (n_atoms <= 60)
    abs_sym = Element.from_Z(int(atomic_numbers[args.absorber_idx])).symbol

    # Edges only
    setup_axis(
        ax_edges,
        pmg_obj,
        coords,
        atomic_numbers,
        args.absorber_idx,
        vis_points,
        is_periodic,
        f"Edges ({args.graph_method}, E={edge_index.shape[1]})",
        label_atoms,
    )
    n_pbc = plot_edges(ax_edges, coords, edge_src, edge_dst, edge_vec_np, is_periodic)
    edge_legend = [Patch(color=(0.15, 0.45, 0.75, 0.6), label="intra-cell")]
    if is_periodic:
        edge_legend.append(Patch(color=(0.85, 0.30, 0.10, 0.85), label=f"PBC-crossing ({n_pbc})"))
    show_voronoi = args.show_voronoi
    voronoi_facets: list = []
    if show_voronoi:
        voronoi_facets = compute_voronoi_facets(pmg_obj, args.cutoff)
        plot_voronoi_facets(ax_edges, voronoi_facets)
        edge_legend.append(Patch(color=(0.27, 0.00, 0.33, 0.55), label=f"Voronoi facets ({len(voronoi_facets)})"))
    ax_edges.legend(handles=edge_legend, loc="upper left", fontsize=8)

    # Geometrygraph triplets
    setup_axis(
        ax_trip,
        pmg_obj,
        coords,
        atomic_numbers,
        args.absorber_idx,
        vis_points,
        is_periodic,
        f"Triplets (T={idx_kj_np.size}, <={args.max_triplets_drawn} drawn)",
        label_atoms,
    )
    plot_triplets(ax_trip, coords, edge_src, edge_dst, edge_vec_np, idx_kj_np, idx_ji_np, args.max_triplets_drawn)
    ax_trip.legend(
        handles=[Patch(color=(0.30, 0.75, 0.35, 0.35), label="triplet (k-j-i)")], loc="upper left", fontsize=8
    )

    # E3EE absorber paths
    n_paths_total = int(paths["path_j"].numel())
    setup_axis(
        ax_paths,
        pmg_obj,
        coords,
        atomic_numbers,
        args.absorber_idx,
        vis_points,
        is_periodic,
        f"Absorber paths from {abs_sym}{args.absorber_idx} " f"(P={n_paths_total}, <={args.max_paths_drawn} drawn)",
        label_atoms,
    )
    plot_absorber_paths(ax_paths, pmg_obj, args.absorber_idx, args.cutoff, args.max_paths, args.max_paths_drawn)
    ax_paths.legend(
        handles=[Patch(color=(0.85, 0.25, 0.55, 0.35), label="(absorber, j, k)")], loc="upper left", fontsize=8
    )

    # Histograms
    ax_hist_w.hist(edge_w_np, bins=30, color="steelblue", edgecolor="white")
    ax_hist_w.set_title("edge weights [A]")
    ax_hist_w.set_xlabel("distance")
    ax_hist_w.set_ylabel("count")
    ax_hist_w.axvline(args.cutoff, color="red", ls="--", lw=1.0, label=f"cutoff={args.cutoff}")
    if edge_attr is not None and edge_attr.numel() > 0:
        ax_twin = ax_hist_w.twinx()
        ax_twin.hist(
            edge_attr.numpy(), bins=30, color="#d98a1d", edgecolor="white", alpha=0.45, label="facet area [A^2]"
        )
        ax_twin.set_ylabel("facet area count", color="#d98a1d")
        ax_twin.tick_params(axis="y", labelcolor="#d98a1d")
    ax_hist_w.legend(fontsize=8)

    deg = np.bincount(edge_src, minlength=n_atoms)
    ax_hist_deg.bar(np.arange(n_atoms), deg, color="slategray", edgecolor="white")
    ax_hist_deg.axhline(args.max_neighbors, color="red", ls="--", lw=1.0, label=f"max_neighbors={args.max_neighbors}")
    ax_hist_deg.set_title(f"per-atom out-degree  (saturated: " f"{int((deg >= args.max_neighbors).sum())}/{n_atoms})")
    ax_hist_deg.set_xlabel("atom index")
    ax_hist_deg.set_ylabel("num edges")
    isolated = np.where(deg == 0)[0]
    if isolated.size > 0:
        iso_str = ",".join(str(int(i)) for i in isolated[:10])
        if isolated.size > 10:
            iso_str += ",..."
        ax_hist_deg.text(
            0.5,
            0.92,
            f"WARNING: {isolated.size} isolated atom(s): [{iso_str}]",
            transform=ax_hist_deg.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="white",
            bbox={"facecolor": "crimson", "alpha": 0.85, "edgecolor": "none", "pad": 3.0},
        )
    ax_hist_deg.legend(fontsize=8, loc="upper right")

    has_trip = angle_np.size > 0
    has_path = n_paths_total > 0
    if has_trip:
        ax_hist_ang.hist(
            np.rad2deg(angle_np),
            bins=36,
            color="seagreen",
            edgecolor="white",
            alpha=0.75,
            label=f"triplet (T={angle_np.size})",
        )
    if has_path:
        cos_paths = paths["path_cosangle"].numpy()
        path_ang = np.rad2deg(np.arccos(np.clip(cos_paths, -1.0, 1.0)))
        ax_hist_ang.hist(
            path_ang, bins=36, color="#c2185b", edgecolor="white", alpha=0.7, label=f"abs-path (P={path_ang.size})"
        )
    if has_trip or has_path:
        ax_hist_ang.set_title("angle distributions [deg]")
        ax_hist_ang.set_xlabel("angle")
        ax_hist_ang.legend(fontsize=8)
    else:
        ax_hist_ang.text(0.5, 0.5, "no triplets / paths", ha="center", va="center", transform=ax_hist_ang.transAxes)
        ax_hist_ang.set_axis_off()

    # Stdout summary
    print("=" * 66)
    print(f"datasource:      {args.json_dir}")
    print(f"sample:          {stem}")
    print(f"kind:            {'periodic Structure' if is_periodic else 'Molecule'}")
    print(f"# atoms:         {n_atoms}")
    print(f"absorber:        idx={args.absorber_idx}  ({abs_sym})")
    print(f"cutoff:          {args.cutoff} A   max_neighbors: {args.max_neighbors}")
    print(f"graph method:    {args.graph_method}")
    if args.graph_method == "voronoi":
        print(f"min facet area:  {args.min_facet_area}")
    if args.graph_method == "cov_radius":
        print(f"cov radii scale: {args.cov_radii_scale}")
    print(f"# edges:         {edge_index.shape[1]}  PBC crossings: {n_pbc}")
    if edge_w_np.size:
        print(f"edge weight:     min={edge_w_np.min():.3f}  " f"mean={edge_w_np.mean():.3f}  max={edge_w_np.max():.3f}")
    if edge_attr is not None and edge_attr.numel() > 0:
        ea = edge_attr.numpy()
        n_zero = int((ea < 1e-6).sum())
        print(
            f"facet area:      min={ea.min():.3g}  mean={ea.mean():.3g}  max={ea.max():.3g}  "
            f"near-zero(<1e-6): {n_zero}/{ea.size}"
        )
    if isolated.size > 0:
        print(f"!! WARNING: {isolated.size} isolated atom(s) (deg=0): " f"{isolated.tolist()}")
    print(
        f"out-degree:      min={int(deg.min())}  "
        f"mean={deg.mean():.2f}  max={int(deg.max())}  "
        f"saturated: {int((deg >= args.max_neighbors).sum())}"
    )
    if has_trip:
        da = np.rad2deg(angle_np)
        n_zero = int((da < 1e-3).sum())
        print(
            f"# triplets:      {angle_np.size}   "
            f"angle(deg) min={da.min():.1f} mean={da.mean():.1f} max={da.max():.1f}"
        )
        if n_zero > 0:
            print(
                f"  ! {n_zero} triplets with angle ~= 0" f"  (periodic bounce-back: likely a bug in triplets.py filter)"
            )
    if has_path:
        print(f"# absorber paths (<={args.max_paths}): {n_paths_total}")
        print(f"  r0j: min={paths['path_r0j'].min():.3f} " f"max={paths['path_r0j'].max():.3f}")
        print(f"  rjk: min={paths['path_rjk'].min():.3f} " f"max={paths['path_rjk'].max():.3f}")
    print("=" * 66)

    fig.suptitle(
        f"{'periodic Structure' if is_periodic else 'Molecule'}  .  {stem}  "
        f".  absorber={abs_sym}{args.absorber_idx}  "
        f".  method={args.graph_method}  "
        f".  cutoff={args.cutoff}  .  max_nbrs={args.max_neighbors}",
        fontsize=11,
    )
    # GridSpec already handles layout; skip tight_layout to avoid clipping 3D axes.

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Figure saved to {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
