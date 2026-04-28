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

"""Visualize E3EE encoder and attention graph diagnostics for PMGJSON samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pymatgen.core import Element

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from graph_tester import (
    compute_voronoi_facets,
    load_sample,
    plot_edges,
    plot_voronoi_facets,
    setup_axis,
)

from xanesnet.utils.graph import GRAPH_METHODS, build_edges

VoronoiFacet: TypeAlias = tuple[np.ndarray, float, float]


def _coerce_min_facet_area(val: str | None) -> float | str | None:
    """Return a numeric or percentage Voronoi facet-area threshold.

    Args:
        val: Raw command-line value. Empty strings and ``None`` disable filtering.

    Returns:
        ``None``, a float threshold in **angstrom squared**, or a percentage string.
    """
    if val is None or val == "":
        return None
    if val.endswith("%"):
        return val
    return float(val)


def _draw_directed_edges(
    ax: Any,
    coords: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_vec: np.ndarray,
    is_periodic: bool,
    color_intra: tuple[float, float, float, float],
    color_pbc: tuple[float, float, float, float],
    width_intra: float = 1.2,
    width_pbc: float = 0.5,
) -> int:
    """Plot directed graph edges on a 3D axis.

    Args:
        ax: Matplotlib 3D axis to draw on.
        coords: Cartesian atom coordinates with shape ``(N, 3)`` in **angstrom**.
        edge_src: Source atom indices with shape ``(E,)``.
        edge_dst: Destination atom indices with shape ``(E,)``.
        edge_vec: Edge vectors with shape ``(E, 3)`` in **angstrom**.
        is_periodic: Whether periodic boundary crossings should be highlighted.
        color_intra: RGBA colour for intra-cell edges.
        color_pbc: RGBA colour for periodic-boundary edges.
        width_intra: Line width for intra-cell edges.
        width_pbc: Line width for periodic-boundary edges.

    Returns:
        Number of edges whose drawn endpoint crosses a periodic boundary.
    """
    segments: list[np.ndarray] = []
    colors: list[tuple[float, float, float, float]] = []
    widths: list[float] = []
    n_pbc = 0
    for s, d, vec in zip(edge_src.tolist(), edge_dst.tolist(), edge_vec):
        start = coords[s]
        end = start + np.asarray(vec, dtype=np.float64)
        segments.append(np.stack([start, end], axis=0))
        if is_periodic:
            delta = float(np.linalg.norm(end - coords[d]))
            if delta > 1e-6:
                colors.append(color_pbc)
                widths.append(width_pbc)
                n_pbc += 1
            else:
                colors.append(color_intra)
                widths.append(width_intra)
        else:
            colors.append(color_intra)
            widths.append(width_intra)
    if segments:
        ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=widths))
    return n_pbc


def _absorber_slice(
    att_edge_index: torch.Tensor,
    att_edge_weight: torch.Tensor,
    att_edge_vec: torch.Tensor,
    absorber_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the absorber-sourced attention subgraph with a self-loop prepended.

    Args:
        att_edge_index: Attention edge indices with shape ``(2, E)``.
        att_edge_weight: Attention edge distances with shape ``(E,)`` in **angstrom**.
        att_edge_vec: Attention edge vectors with shape ``(E, 3)`` in **angstrom**.
        absorber_idx: Atom index used as the attention query source.

    Returns:
        Source indices, destination indices, distances, and vectors for the absorber slice.
        The first entry is the zero-distance self-loop used by ``E3EEDataset.prepare``.
    """
    src = att_edge_index[0]
    dst = att_edge_index[1]
    sel = src == absorber_idx
    dst_site = torch.cat(
        [
            torch.tensor([absorber_idx], dtype=torch.int64),
            dst[sel].to(dtype=torch.int64),
        ],
        dim=0,
    )
    src_site = torch.full_like(dst_site, absorber_idx)
    weight_site = torch.cat(
        [
            torch.zeros(1, dtype=torch.float32),
            att_edge_weight[sel].to(dtype=torch.float32),
        ],
        dim=0,
    )
    vec_site = torch.cat(
        [
            torch.zeros(1, 3, dtype=torch.float32),
            att_edge_vec[sel].to(dtype=torch.float32),
        ],
        dim=0,
    )
    return (
        src_site.numpy(),
        dst_site.numpy(),
        weight_site.numpy(),
        vec_site.numpy(),
    )


def main() -> None:
    """Run the E3EE graph diagnostic command-line interface."""

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json-dir", required=True, type=Path)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--index", type=int, help="Sample index into sorted datasource")
    g.add_argument("--file", type=str, help="Sample file stem (no .json)")

    # Main graph params (encoder graph).
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--max-neighbors", type=int, default=32)
    p.add_argument("--graph-method", type=str, default="radius", choices=list(GRAPH_METHODS))
    p.add_argument("--min-facet-area", type=str, default=None)
    p.add_argument("--cov-radii-scale", type=float, default=1.5)
    p.add_argument("--show-voronoi", action="store_true", help="Overlay Voronoi facets on the main edges panel")

    # Attention graph params.
    p.add_argument("--att-cutoff", type=float, default=10.0)
    p.add_argument("--att-max-neighbors", type=int, default=64)
    p.add_argument("--att-graph-method", type=str, default="radius", choices=list(GRAPH_METHODS))
    p.add_argument("--att-min-facet-area", type=str, default=None)
    p.add_argument("--att-cov-radii-scale", type=float, default=1.5)

    # Drawing / output params.
    p.add_argument("--absorber-idx", type=int, default=0)
    p.add_argument("--no-atom-labels", action="store_true")
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    pmg_obj = load_sample(args.json_dir, args.index, args.file)
    stem = pmg_obj.properties.get("file_name", "<unknown>")
    is_periodic = hasattr(pmg_obj, "lattice")
    coords = np.array(pmg_obj.cart_coords, dtype=np.float64)
    atomic_numbers = np.array(pmg_obj.atomic_numbers, dtype=np.int64)
    n_atoms = len(pmg_obj)
    if not (0 <= args.absorber_idx < n_atoms):
        raise SystemExit(f"--absorber-idx {args.absorber_idx} out of range [0, {n_atoms})")

    main_mfa = _coerce_min_facet_area(args.min_facet_area)
    att_mfa = _coerce_min_facet_area(args.att_min_facet_area)

    edge_index, edge_weight, edge_vec, edge_attr = build_edges(
        pmg_obj,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_neighbors,
        compute_vectors=True,
        method=args.graph_method,
        min_facet_area=main_mfa,
        cov_radii_scale=args.cov_radii_scale,
    )
    assert edge_vec is not None
    edge_src = edge_index[0].numpy()
    edge_dst = edge_index[1].numpy()
    edge_vec_np = edge_vec.numpy()
    edge_w_np = edge_weight.numpy()

    att_edge_index, att_edge_weight, att_edge_vec, _ = build_edges(
        pmg_obj,
        cutoff=args.att_cutoff,
        max_num_neighbors=args.att_max_neighbors,
        compute_vectors=True,
        method=args.att_graph_method,
        min_facet_area=att_mfa,
        cov_radii_scale=args.att_cov_radii_scale,
    )
    assert att_edge_vec is not None
    att_src_np = att_edge_index[0].numpy()
    att_dst_np = att_edge_index[1].numpy()
    att_vec_np = att_edge_vec.numpy()
    att_w_np = att_edge_weight.numpy()

    abs_src_np, abs_dst_np, abs_w_np, abs_vec_np = _absorber_slice(
        att_edge_index, att_edge_weight, att_edge_vec, args.absorber_idx
    )

    vis_points = coords.copy()
    if is_periodic:
        from graph_tester import _cell_corner_coords  # type: ignore[attr-defined]

        vis_points = np.vstack([vis_points, _cell_corner_coords(pmg_obj)])
    if edge_vec_np.shape[0] > 0:
        vis_points = np.vstack([vis_points, coords[edge_src] + edge_vec_np])
    if att_vec_np.shape[0] > 0:
        vis_points = np.vstack([vis_points, coords[att_src_np] + att_vec_np])

    label_atoms = (not args.no_atom_labels) and (n_atoms <= 60)
    abs_sym = Element.from_Z(int(atomic_numbers[args.absorber_idx])).symbol

    fig = plt.figure(figsize=(20, 14))
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
    ax_main = fig.add_subplot(gs[0, 0], projection="3d")
    ax_att = fig.add_subplot(gs[0, 1], projection="3d")
    ax_abs = fig.add_subplot(gs[0, 2], projection="3d")
    ax_hist_main = fig.add_subplot(gs[1, 0])
    ax_hist_att = fig.add_subplot(gs[1, 1])
    ax_hist_deg = fig.add_subplot(gs[1, 2])

    setup_axis(
        ax_main,
        pmg_obj,
        coords,
        atomic_numbers,
        args.absorber_idx,
        vis_points,
        is_periodic,
        f"Main graph ({args.graph_method}, E={edge_index.shape[1]})",
        label_atoms,
    )
    n_pbc_main = plot_edges(ax_main, coords, edge_src, edge_dst, edge_vec_np, is_periodic)
    main_legend = [Patch(color=(0.15, 0.45, 0.75, 0.6), label="intra-cell")]
    if is_periodic:
        main_legend.append(Patch(color=(0.85, 0.30, 0.10, 0.55), label="PBC crossing"))
    voronoi_facets: list[VoronoiFacet] = []
    if args.show_voronoi:
        voronoi_facets = compute_voronoi_facets(pmg_obj, args.cutoff)
        plot_voronoi_facets(ax_main, voronoi_facets)
        main_legend.append(Patch(color=(0.30, 0.45, 0.75, 0.30), label="Voronoi facet"))
    ax_main.legend(handles=main_legend, loc="upper left", fontsize=8)

    setup_axis(
        ax_att,
        pmg_obj,
        coords,
        atomic_numbers,
        args.absorber_idx,
        vis_points,
        is_periodic,
        f"Attention graph ({args.att_graph_method}, E={att_edge_index.shape[1]})",
        label_atoms,
    )
    n_pbc_att = _draw_directed_edges(
        ax_att,
        coords,
        att_src_np,
        att_dst_np,
        att_vec_np,
        is_periodic,
        color_intra=(0.55, 0.20, 0.65, 0.55),
        color_pbc=(0.85, 0.30, 0.10, 0.45),
        width_intra=1.0,
        width_pbc=0.4,
    )
    att_legend = [Patch(color=(0.55, 0.20, 0.65, 0.55), label="att edge (intra)")]
    if is_periodic:
        att_legend.append(Patch(color=(0.85, 0.30, 0.10, 0.45), label="att edge (PBC)"))
    ax_att.legend(handles=att_legend, loc="upper left", fontsize=8)

    n_abs_edges = abs_dst_np.size
    setup_axis(
        ax_abs,
        pmg_obj,
        coords,
        atomic_numbers,
        args.absorber_idx,
        vis_points,
        is_periodic,
        f"Absorber slice from {abs_sym}{args.absorber_idx} " f"(E={n_abs_edges}, +1 self-loop)",
        label_atoms,
    )
    has_vec = np.linalg.norm(abs_vec_np, axis=-1) > 1e-9
    n_pbc_abs = _draw_directed_edges(
        ax_abs,
        coords,
        abs_src_np[has_vec],
        abs_dst_np[has_vec],
        abs_vec_np[has_vec],
        is_periodic,
        color_intra=(0.85, 0.25, 0.55, 0.85),
        color_pbc=(0.85, 0.30, 0.10, 0.55),
        width_intra=1.6,
        width_pbc=0.6,
    )
    ax_abs.scatter(
        coords[args.absorber_idx, 0],
        coords[args.absorber_idx, 1],
        coords[args.absorber_idx, 2],
        s=320,
        facecolors="none",
        edgecolors=(0.85, 0.25, 0.55, 0.9),
        linewidths=1.5,
    )
    abs_legend = [
        Patch(color=(0.85, 0.25, 0.55, 0.85), label="absorber -> key"),
        Patch(color="none", label=f"self-loop ({abs_sym}{args.absorber_idx})"),
    ]
    if is_periodic:
        abs_legend.insert(1, Patch(color=(0.85, 0.30, 0.10, 0.55), label="PBC crossing"))
    ax_abs.legend(handles=abs_legend, loc="upper left", fontsize=8)

    if edge_w_np.size:
        ax_hist_main.hist(edge_w_np, bins=30, color="steelblue", edgecolor="white")
    ax_hist_main.axvline(args.cutoff, color="red", ls="--", lw=1.0, label=f"cutoff={args.cutoff}")
    ax_hist_main.set_title("main edge weights [A]")
    ax_hist_main.set_xlabel("distance")
    ax_hist_main.set_ylabel("count")
    ax_hist_main.legend(fontsize=8)

    if att_w_np.size:
        ax_hist_att.hist(att_w_np, bins=30, color=(0.55, 0.20, 0.65), edgecolor="white")
    ax_hist_att.axvline(args.att_cutoff, color="red", ls="--", lw=1.0, label=f"att_cutoff={args.att_cutoff}")
    ax_hist_att.set_title("attention edge weights [A]")
    ax_hist_att.set_xlabel("distance")
    ax_hist_att.set_ylabel("count")
    ax_hist_att.legend(fontsize=8)

    att_deg = np.bincount(att_src_np, minlength=n_atoms) if att_src_np.size else np.zeros(n_atoms, dtype=np.int64)
    ax_hist_deg.bar(np.arange(n_atoms), att_deg, color=(0.55, 0.20, 0.65), edgecolor="white")
    ax_hist_deg.axhline(
        args.att_max_neighbors,
        color="red",
        ls="--",
        lw=1.0,
        label=f"att_max_neighbors={args.att_max_neighbors}",
    )
    ax_hist_deg.set_title(f"att out-degree (saturated: {int((att_deg >= args.att_max_neighbors).sum())}/{n_atoms})")
    ax_hist_deg.set_xlabel("atom index")
    ax_hist_deg.set_ylabel("# att edges (src)")
    isolated_att = np.where(att_deg == 0)[0]
    if isolated_att.size > 0:
        ax_hist_deg.text(
            0.98,
            0.95,
            f"!! {isolated_att.size} isolated atom(s) in att graph",
            ha="right",
            va="top",
            transform=ax_hist_deg.transAxes,
            fontsize=8,
            color="red",
        )
    ax_hist_deg.legend(fontsize=8, loc="upper right")

    main_deg = np.bincount(edge_src, minlength=n_atoms) if edge_src.size else np.zeros(n_atoms, dtype=np.int64)
    isolated_main = np.where(main_deg == 0)[0]
    print("=" * 66)
    print(f"datasource:      {args.json_dir}")
    print(f"sample:          {stem}")
    print(f"kind:            {'periodic Structure' if is_periodic else 'Molecule'}")
    print(f"# atoms:         {n_atoms}")
    print(f"absorber:        idx={args.absorber_idx}  ({abs_sym})")
    print("-" * 66)
    print(f"MAIN  cutoff={args.cutoff}  max_nbrs={args.max_neighbors}  method={args.graph_method}")
    print(f"  # edges: {edge_index.shape[1]}  PBC crossings: {n_pbc_main}")
    if edge_w_np.size:
        print(f"  weight: min={edge_w_np.min():.3f}  mean={edge_w_np.mean():.3f}  max={edge_w_np.max():.3f}")
    print(
        f"  out-deg: min={int(main_deg.min())}  mean={main_deg.mean():.2f}  "
        f"max={int(main_deg.max())}  saturated: {int((main_deg >= args.max_neighbors).sum())}"
    )
    if isolated_main.size > 0:
        print(f"  !! {isolated_main.size} isolated atom(s): {isolated_main.tolist()}")
    print("-" * 66)
    print(f"ATT   cutoff={args.att_cutoff}  max_nbrs={args.att_max_neighbors}  method={args.att_graph_method}")
    print(f"  # edges: {att_edge_index.shape[1]}  PBC crossings: {n_pbc_att}")
    if att_w_np.size:
        print(f"  weight: min={att_w_np.min():.3f}  mean={att_w_np.mean():.3f}  max={att_w_np.max():.3f}")
    print(
        f"  out-deg: min={int(att_deg.min())}  mean={att_deg.mean():.2f}  "
        f"max={int(att_deg.max())}  saturated: {int((att_deg >= args.att_max_neighbors).sum())}"
    )
    if isolated_att.size > 0:
        print(f"  !! {isolated_att.size} isolated atom(s) in att graph: {isolated_att.tolist()}")
    print("-" * 66)
    print(
        f"absorber slice ({abs_sym}{args.absorber_idx}): "
        f"{int(has_vec.sum())} edges + 1 self-loop  (PBC: {n_pbc_abs})"
    )
    print("=" * 66)

    fig.suptitle(
        f"{'periodic Structure' if is_periodic else 'Molecule'}  .  {stem}  "
        f".  absorber={abs_sym}{args.absorber_idx}  "
        f".  main: {args.graph_method}/{args.cutoff}A  "
        f".  att: {args.att_graph_method}/{args.att_cutoff}A",
        fontsize=11,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Figure saved to {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
