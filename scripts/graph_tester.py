"""
Graph construction visual tester.

Standalone diagnostic script to visually confirm that the graph-building logic
used by RadiusGraphDataset and E3EEDataset behaves correctly for both periodic
Structures and non-periodic Molecules. Useful when picking sensible values for
``cutoff`` and ``max_num_neighbors``.

Input is a ``pmgjson`` datasource directory (same layout as used by the rest
of XANESNET) - each ``.json`` holds a single pymatgen Structure or Molecule.
A sample is selected by index or by file stem.

Produces three 3D views of the same structure - edges only, radiusgraph
triplets only, e3ee absorber paths only - plus histograms of edge weights,
per-atom out-degree, and angle distributions.

Run:
    python scripts/graph_tester.py --json-dir data/fe/json_train --index 0 \
        --cutoff 6.0 --max-neighbors 32

This script does NOT import any model code; it only exercises the shared
``xanesnet.utils.graph`` helpers, matching what the datasets do at prepare().
"""

from __future__ import annotations

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

from xanesnet.datasources.pmgjson import PMGJSONSource  # noqa: E402
from xanesnet.utils.graph import (  # noqa: E402
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
        except Exception:  # noqa: BLE001
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
    """Draw intra-cell edges boldly; PBC-crossing edges thinner and more faded
    so focus stays on the primary graph."""
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

    edge_index, edge_weight, edge_vec = build_edges(
        pmg_obj,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_neighbors,
        compute_vectors=True,
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

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[3.0, 1.0],
        hspace=0.18,
        wspace=0.12,
        left=0.03,
        right=0.99,
        top=0.94,
        bottom=0.06,
    )
    ax_edges = fig.add_subplot(gs[0, 0], projection="3d")
    ax_trip = fig.add_subplot(gs[0, 1], projection="3d")
    ax_paths = fig.add_subplot(gs[0, 2], projection="3d")
    ax_hist_w = fig.add_subplot(gs[1, 0])
    ax_hist_deg = fig.add_subplot(gs[1, 1])
    ax_hist_ang = fig.add_subplot(gs[1, 2])

    label_atoms = (not args.no_atom_labels) and (n_atoms <= 60)
    abs_sym = Element.from_Z(int(atomic_numbers[args.absorber_idx])).symbol

    # --- Edges only
    setup_axis(
        ax_edges,
        pmg_obj,
        coords,
        atomic_numbers,
        args.absorber_idx,
        vis_points,
        is_periodic,
        f"Edges (E={edge_index.shape[1]})",
        label_atoms,
    )
    n_pbc = plot_edges(ax_edges, coords, edge_src, edge_dst, edge_vec_np, is_periodic)
    edge_legend = [Patch(color=(0.15, 0.45, 0.75, 0.6), label="intra-cell")]
    if is_periodic:
        edge_legend.append(Patch(color=(0.85, 0.30, 0.10, 0.85), label=f"PBC-crossing ({n_pbc})"))
    ax_edges.legend(handles=edge_legend, loc="upper left", fontsize=8)

    # --- Radiusgraph triplets
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

    # --- E3EE absorber paths
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

    # --- Histograms
    ax_hist_w.hist(edge_w_np, bins=30, color="steelblue", edgecolor="white")
    ax_hist_w.set_title("edge weights [A]")
    ax_hist_w.set_xlabel("distance")
    ax_hist_w.set_ylabel("count")
    ax_hist_w.axvline(args.cutoff, color="red", ls="--", lw=1.0, label=f"cutoff={args.cutoff}")
    ax_hist_w.legend(fontsize=8)

    deg = np.bincount(edge_src, minlength=n_atoms)
    ax_hist_deg.bar(np.arange(n_atoms), deg, color="slategray", edgecolor="white")
    ax_hist_deg.axhline(args.max_neighbors, color="red", ls="--", lw=1.0, label=f"max_neighbors={args.max_neighbors}")
    ax_hist_deg.set_title(f"per-atom out-degree  (saturated: " f"{int((deg >= args.max_neighbors).sum())}/{n_atoms})")
    ax_hist_deg.set_xlabel("atom index")
    ax_hist_deg.set_ylabel("num edges")
    ax_hist_deg.legend(fontsize=8)

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

    # --- Stdout summary
    print("=" * 66)
    print(f"datasource:      {args.json_dir}")
    print(f"sample:          {stem}")
    print(f"kind:            {'periodic Structure' if is_periodic else 'Molecule'}")
    print(f"# atoms:         {n_atoms}")
    print(f"absorber:        idx={args.absorber_idx}  ({abs_sym})")
    print(f"cutoff:          {args.cutoff} A   max_neighbors: {args.max_neighbors}")
    print(f"# edges:         {edge_index.shape[1]}  PBC crossings: {n_pbc}")
    if edge_w_np.size:
        print(f"edge weight:     min={edge_w_np.min():.3f}  " f"mean={edge_w_np.mean():.3f}  max={edge_w_np.max():.3f}")
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
