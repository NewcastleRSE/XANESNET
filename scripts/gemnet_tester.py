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
GemNet / GemNet-OC graph construction visual tester.

Standalone diagnostic script that mirrors ``scripts/graph_tester.py`` but
exercises every graph / index tensor that ``GemNetDataset`` produces for a
single sample: the main embedding graph, triplets, quadruplets (with their
interaction edges), and - when ``--oc`` is set - the GemNet-OC auxiliary
graphs (``a2ee2a``, ``a2a``, ``qint``) plus the mixed (a2e / e2a) triplets.

Input is a ``pmgjson`` datasource directory - one pymatgen Structure or
Molecule per ``.json``. A sample is picked by index or file stem, exactly
like ``graph_tester.py``.

3D views (top row):
    1. Main edges      (+ optional Voronoi facet overlay)
    2. Triplets        (k-j-i on the main graph)
    3. Quadruplets     (c-a-b-d along the interaction graph) OR absorber
                        mixed triplets if ``--oc`` and quadruplets disabled.

If ``--oc`` is set, a second 3D row is drawn:
    4. a2ee2a edges
    5. a2a edges
    6. qint edges (main+int overlap)

Histograms (bottom row):
    - edge weight distributions for each active graph (stacked)
    - per-atom out-degree for each active graph
    - triplet angles + quadruplet dihedrals + (OC) mixed triplet angles

Stdout prints a full per-graph summary plus id_swap involution / index-
range sanity checks.

Run examples:
    python scripts/gemnet_tester.py --json-dir data/fe/xanes_train \\
        --index 0 --cutoff 6.0 --max-neighbors 32 --int-cutoff 10.0 \\
        --quadruplets

    python scripts/gemnet_tester.py --json-dir data/omnixas/... \\
        --index 3 --cutoff 5.0 --max-neighbors 50 --int-cutoff 10.0 \\
        --oc --oc-cutoff-aeaint 5.0 --oc-cutoff-aint 10.0 \\
        --graph-method voronoi --show-voronoi --quadruplets

This script does not import any model code, only the shared
``xanesnet.utils.graph`` helpers.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from pymatgen.core import Element, Structure

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Re-use the visual helpers from graph_tester. These are safe to import: the
# argparse-driven main() is guarded by __name__ == "__main__".
from graph_tester import (  # type: ignore  # noqa: E402
    _cell_corner_coords,
    compute_voronoi_facets,
    equalize_3d_axes,
    load_sample,
    plot_edges,
    plot_voronoi_facets,
    setup_axis,
)

from xanesnet.utils.graph import GRAPH_METHODS, build_edges  # noqa: E402
from xanesnet.utils.graph.gemnet_indices import (  # noqa: E402
    compute_id_swap,
    compute_mixed_triplets,
    compute_quadruplets,
    compute_triplets,
)

###############################################################################
############################### small helpers #################################
###############################################################################


def build_main_edges(pmg_obj, cutoff, max_nbrs, method, min_facet_area, cov_radii_scale):
    edge_index, edge_weight, edge_vec, edge_attr = build_edges(
        pmg_obj,
        cutoff=cutoff,
        max_num_neighbors=max_nbrs,
        compute_vectors=True,
        method=method,
        min_facet_area=min_facet_area,
        cov_radii_scale=cov_radii_scale,
    )
    assert edge_vec is not None
    return edge_index, edge_weight, edge_vec, edge_attr


def sample_rng_indices(n: int, k: int, seed: int = 0) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(k, n)
    return np.random.default_rng(seed).choice(n, size=k, replace=False)


def compute_dihedrals(
    edge_vec: torch.Tensor,
    int_edge_vec: torch.Tensor,
    id4_reduce_ca: torch.Tensor,
    id4_expand_db: torch.Tensor,
    id4_int_edge: torch.Tensor,
) -> np.ndarray:
    """
    GemNet dihedral angle per quadruplet c-a-b-d, computed from the three
    vectors vec_ca (main), vec_ba (int), vec_db (main). Returns radians in
    [0, pi].
    """
    if id4_reduce_ca.numel() == 0:
        return np.zeros(0, dtype=np.float64)
    vec_ca = edge_vec[id4_reduce_ca]  # pos_a - pos_c
    vec_ba = int_edge_vec[id4_int_edge]  # pos_a - pos_b
    vec_db = edge_vec[id4_expand_db]  # pos_b - pos_d
    # Reconstruct path vectors along c -> a -> b -> d
    b1 = vec_ca  # a - c
    b2 = -vec_ba  # b - a
    b3 = -vec_db  # d - b
    n1 = torch.linalg.cross(b1, b2, dim=-1)
    n2 = torch.linalg.cross(b2, b3, dim=-1)
    b2n = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-12)
    m1 = torch.linalg.cross(n1, b2n, dim=-1)
    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    dih = torch.atan2(y, x).abs()  # map to [0, pi]
    return dih.detach().cpu().numpy()


def quad_int_edge_from_quad(
    n_int: int,
    int_edge_index: torch.Tensor,
    edge_index: torch.Tensor,
    id4_reduce_ca: torch.Tensor,
    id4_expand_db: torch.Tensor,
) -> torch.Tensor:
    """
    Recover, per quadruplet, the id of the interaction edge b->a. We know
    ``a = main_dst[id4_reduce_ca]`` and ``b = main_src[id4_expand_db]``, and
    that the int edge has source=b, target=a. Since the same (b,a) pair can
    exist in the int graph only once per periodic image, we match via
    int_edge_index columns. This is only used for diagnostic dihedrals so we
    accept a linear fallback via a dict lookup when needed.
    """
    if id4_reduce_ca.numel() == 0:
        return torch.empty(0, dtype=torch.int64)
    a = edge_index[1][id4_reduce_ca]
    b = edge_index[0][id4_expand_db]  # src of d->b is d, dst is b -> index 1
    # NOTE: in GemNetData we store edge c->a with source=c,target=a, and
    # edge d->b with source=d,target=b. So b = dst of id4_expand_db.
    b = edge_index[1][id4_expand_db]
    idx_int_s = int_edge_index[0]  # b
    idx_int_t = int_edge_index[1]  # a
    # Build dictionary (b, a) -> first int edge id. For periodic, multiple
    # images per (b,a) are OK: the dihedral is approximate when there is no
    # 1:1 correspondence (this script is a visual diagnostic, not training).
    lookup: dict[tuple[int, int], int] = {}
    for i in range(n_int):
        lookup.setdefault((int(idx_int_s[i].item()), int(idx_int_t[i].item())), i)
    out = torch.empty(id4_reduce_ca.numel(), dtype=torch.int64)
    for k in range(id4_reduce_ca.numel()):
        key = (int(b[k].item()), int(a[k].item()))
        out[k] = lookup.get(key, -1)
    return out


###############################################################################
############################### 3D plotters ###################################
###############################################################################


def _draw_generic_edges(ax, coords, edge_src, edge_dst, edge_vec, is_periodic, color_intra, color_pbc, alpha=0.75):
    segments = []
    colors = []
    widths = []
    n_pbc = 0
    for s, d, vec in zip(edge_src.tolist(), edge_dst.tolist(), edge_vec):
        start = coords[s]
        end = start + np.asarray(vec, dtype=np.float64)
        segments.append(np.stack([start, end], axis=0))
        if is_periodic and np.linalg.norm(end - coords[d]) > 1e-6:
            colors.append((*color_pbc, 0.55))
            widths.append(0.5)
            n_pbc += 1
        else:
            colors.append((*color_intra, alpha))
            widths.append(1.2)
    ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=widths))
    return n_pbc


def plot_triplets(ax, coords, edge_src, edge_dst, edge_vec, id3_reduce_ca, id3_expand_ba, max_draw, seed=0):
    n = id3_reduce_ca.shape[0]
    if n == 0:
        return
    take = sample_rng_indices(n, max_draw, seed=seed)
    tris = []
    for t in take:
        e_ca = int(id3_reduce_ca[t])
        e_ba = int(id3_expand_ba[t])
        a = int(edge_dst[e_ca])
        pa = coords[a]
        pc = pa - np.asarray(edge_vec[e_ca], dtype=np.float64)
        pb = pa - np.asarray(edge_vec[e_ba], dtype=np.float64)
        tris.append(np.stack([pc, pa, pb], axis=0))
    poly = Poly3DCollection(
        tris, facecolors=(0.30, 0.75, 0.35, 0.22), edgecolors=(0.10, 0.50, 0.15, 0.7), linewidths=0.7
    )
    ax.add_collection3d(poly)


def plot_quadruplets(
    ax,
    coords,
    edge_src,
    edge_dst,
    edge_vec,
    int_edge_src,
    int_edge_dst,
    int_edge_vec,
    id4_reduce_ca,
    id4_expand_db,
    id4_int_edge,
    max_draw,
    seed=0,
):
    n = id4_reduce_ca.shape[0]
    if n == 0:
        return 0
    take = sample_rng_indices(n, max_draw, seed=seed)
    polylines = []
    for t in take:
        e_ca = int(id4_reduce_ca[t])
        e_db = int(id4_expand_db[t])
        e_ba = int(id4_int_edge[t])
        if e_ba < 0:
            continue
        a = int(edge_dst[e_ca])
        pa = coords[a]
        pc = pa - np.asarray(edge_vec[e_ca], dtype=np.float64)
        pb = pa - np.asarray(int_edge_vec[e_ba], dtype=np.float64)
        pd = pb - np.asarray(edge_vec[e_db], dtype=np.float64)
        polylines.append(np.stack([pc, pa, pb, pd], axis=0))
    if not polylines:
        return 0
    segs = []
    for poly in polylines:
        segs.append(poly[0:2])
        segs.append(poly[1:3])
        segs.append(poly[2:4])
    # Color: c-a (green), a-b (orange = int edge), b-d (blue)
    ncolors = [(0.10, 0.55, 0.15, 0.85), (0.95, 0.50, 0.10, 0.85), (0.20, 0.40, 0.80, 0.85)] * len(polylines)
    ax.add_collection3d(Line3DCollection(segs, colors=ncolors, linewidths=1.2))
    return len(polylines)


def plot_mixed_triplets(
    ax,
    coords,
    main_edge_src,
    main_edge_dst,
    main_edge_vec,
    other_edge_src,
    other_edge_vec,
    idx_out,
    idx_in,
    max_draw,
    color,
    seed=0,
):
    n = idx_out.shape[0]
    if n == 0:
        return
    take = sample_rng_indices(n, max_draw, seed=seed)
    tris = []
    for t in take:
        eo = int(idx_out[t])
        ei = int(idx_in[t])
        # Main output edge c->a: target a is pivot for to_outedge=False matches
        a = int(main_edge_dst[eo])
        pa = coords[a]
        pc = pa - np.asarray(main_edge_vec[eo], dtype=np.float64)
        # Input edge source/target on the other graph
        sb = int(other_edge_src[ei])
        pb = coords[sb]
        tris.append(np.stack([pc, pa, pb], axis=0))
    poly = Poly3DCollection(tris, facecolors=(*color, 0.22), edgecolors=(*color, 0.75), linewidths=0.7)
    ax.add_collection3d(poly)


###############################################################################
################################# main ########################################
###############################################################################


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json-dir", required=True, type=Path)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--index", type=int, help="Sample index into sorted datasource")
    g.add_argument("--file", type=str, help="Sample file stem (no .json)")

    # Main graph params
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--max-neighbors", type=int, default=32)
    p.add_argument("--graph-method", type=str, default="radius", choices=list(GRAPH_METHODS))
    p.add_argument("--min-facet-area", type=str, default=None)
    p.add_argument("--cov-radii-scale", type=float, default=1.5)
    p.add_argument("--show-voronoi", action="store_true", help="Overlay Voronoi facets on the main edges panel")

    # Quadruplet / interaction graph params
    p.add_argument("--quadruplets", action="store_true", help="Compute quadruplet indices")
    p.add_argument("--int-cutoff", type=float, default=None, help="Interaction cutoff; defaults to --cutoff")

    # GemNet-OC params
    p.add_argument("--oc", action="store_true", help="Enable GemNet-OC mode (extra graphs + mixed triplets)")
    p.add_argument("--oc-cutoff-aeaint", type=float, default=None)
    p.add_argument("--oc-cutoff-aint", type=float, default=None)
    p.add_argument("--oc-max-neighbors-aeaint", type=int, default=None)
    p.add_argument("--oc-max-neighbors-aint", type=int, default=None)

    # Drawing params
    p.add_argument("--absorber-idx", type=int, default=0)
    p.add_argument("--max-triplets-drawn", type=int, default=60)
    p.add_argument("--max-quads-drawn", type=int, default=60)
    p.add_argument("--max-mixed-drawn", type=int, default=60)
    p.add_argument("--no-atom-labels", action="store_true")
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    # ---------- Load sample ----------
    pmg_obj = load_sample(args.json_dir, args.index, args.file)
    stem = pmg_obj.properties.get("file_name", "<unknown>")
    is_periodic = isinstance(pmg_obj, Structure)
    coords = np.array(pmg_obj.cart_coords, dtype=np.float64)
    atomic_numbers = np.array(pmg_obj.atomic_numbers, dtype=np.int64)
    n_atoms = len(pmg_obj)
    if not (0 <= args.absorber_idx < n_atoms):
        raise SystemExit(f"--absorber-idx {args.absorber_idx} out of range [0, {n_atoms})")

    min_facet_area = args.min_facet_area
    if min_facet_area is not None and not min_facet_area.endswith("%"):
        min_facet_area = float(min_facet_area)

    int_cutoff = args.int_cutoff if args.int_cutoff is not None else args.cutoff
    oc_cutoff_aeaint = args.oc_cutoff_aeaint if args.oc_cutoff_aeaint is not None else args.cutoff
    oc_cutoff_aint = (
        args.oc_cutoff_aint if args.oc_cutoff_aint is not None else max(args.cutoff, oc_cutoff_aeaint, int_cutoff)
    )
    oc_max_nbrs_aeaint = (
        args.oc_max_neighbors_aeaint if args.oc_max_neighbors_aeaint is not None else args.max_neighbors
    )
    oc_max_nbrs_aint = args.oc_max_neighbors_aint if args.oc_max_neighbors_aint is not None else args.max_neighbors

    # ---------- Build graphs ----------
    edge_index, edge_weight, edge_vec, edge_attr = build_main_edges(
        pmg_obj, args.cutoff, args.max_neighbors, args.graph_method, min_facet_area, args.cov_radii_scale
    )
    edge_src = edge_index[0].numpy()
    edge_dst = edge_index[1].numpy()
    edge_vec_np = edge_vec.numpy()
    edge_w_np = edge_weight.numpy()

    # Triplets on main graph
    id3_reduce_ca, id3_expand_ba, Kidx3 = compute_triplets(edge_index, n_atoms)
    triplet_angles = np.zeros(0, dtype=np.float64)
    if id3_reduce_ca.numel() > 0:
        v_ca = edge_vec[id3_reduce_ca]
        v_ba = edge_vec[id3_expand_ba]
        cos = (v_ca * v_ba).sum(dim=-1) / (v_ca.norm(dim=-1) * v_ba.norm(dim=-1) + 1e-12)
        triplet_angles = torch.arccos(cos.clamp(-1.0, 1.0)).numpy()

    # id_swap sanity (will raise if graph not symmetric — useful check)
    id_swap_ok = True
    id_swap_err = ""
    try:
        id_swap = compute_id_swap(edge_index, edge_vec)
        involution = torch.equal(id_swap[id_swap], torch.arange(edge_index.size(1), dtype=id_swap.dtype))
        id_swap_ok = involution
        if not involution:
            id_swap_err = "id_swap is not an involution"
    except Exception as exc:  # pragma: no cover - diagnostic only
        id_swap_ok = False
        id_swap_err = str(exc)

    # Quadruplets
    int_edge_index = torch.empty(2, 0, dtype=torch.int64)
    int_edge_vec = torch.empty(0, 3)
    int_edge_w_np = np.zeros(0)
    quad: dict = {}
    dihedrals = np.zeros(0, dtype=np.float64)
    if args.quadruplets:
        int_edge_index, int_edge_weight, int_edge_vec, _ = build_main_edges(
            pmg_obj, int_cutoff, args.max_neighbors, args.graph_method, min_facet_area, args.cov_radii_scale
        )
        int_edge_w_np = int_edge_weight.numpy()
        if int_edge_index.size(1) > 0 and edge_index.size(1) > 0:
            quad = compute_quadruplets(edge_index, edge_vec, int_edge_index, int_edge_vec, n_atoms)
            id4_int_edge = quad_int_edge_from_quad(
                int_edge_index.size(1),
                int_edge_index,
                edge_index,
                quad["id4_reduce_ca"],
                quad["id4_expand_db"],
            )
            dihedrals = compute_dihedrals(
                edge_vec, int_edge_vec, quad["id4_reduce_ca"], quad["id4_expand_db"], id4_int_edge
            )
            quad["id4_int_edge_diag"] = id4_int_edge

    # OC graphs
    a2ee2a_edge_index = torch.empty(2, 0, dtype=torch.int64)
    a2ee2a_edge_vec = torch.empty(0, 3)
    a2ee2a_edge_w_np = np.zeros(0)
    a2a_edge_index = torch.empty(2, 0, dtype=torch.int64)
    a2a_edge_vec = torch.empty(0, 3)
    a2a_edge_w_np = np.zeros(0)
    a2e_mixed: dict = {}
    e2a_mixed: dict = {}
    a2e_angles = np.zeros(0, dtype=np.float64)
    e2a_angles = np.zeros(0, dtype=np.float64)
    if args.oc:
        a2ee2a_edge_index, a2ee2a_edge_weight, a2ee2a_edge_vec, _ = build_main_edges(
            pmg_obj, oc_cutoff_aeaint, oc_max_nbrs_aeaint, args.graph_method, min_facet_area, args.cov_radii_scale
        )
        a2ee2a_edge_w_np = a2ee2a_edge_weight.numpy()
        a2a_edge_index, a2a_edge_weight, a2a_edge_vec, _ = build_main_edges(
            pmg_obj, oc_cutoff_aint, oc_max_nbrs_aint, args.graph_method, min_facet_area, args.cov_radii_scale
        )
        a2a_edge_w_np = a2a_edge_weight.numpy()
        if edge_index.size(1) > 0 and a2ee2a_edge_index.size(1) > 0:
            a2e_mixed = compute_mixed_triplets(
                main_edge_index=edge_index,
                main_edge_vec=edge_vec,
                other_edge_index=a2ee2a_edge_index,
                other_edge_vec=a2ee2a_edge_vec,
                num_nodes=n_atoms,
                to_outedge=False,
            )
            e2a_mixed = compute_mixed_triplets(
                main_edge_index=a2ee2a_edge_index,
                main_edge_vec=a2ee2a_edge_vec,
                other_edge_index=edge_index,
                other_edge_vec=edge_vec,
                num_nodes=n_atoms,
                to_outedge=False,
            )
            # Mixed-triplet angles
            if a2e_mixed["in_"].numel() > 0:
                v_out = edge_vec[a2e_mixed["out"]]
                v_in = a2ee2a_edge_vec[a2e_mixed["in_"]]
                cos = (v_out * v_in).sum(dim=-1) / (v_out.norm(dim=-1) * v_in.norm(dim=-1) + 1e-12)
                a2e_angles = torch.arccos(cos.clamp(-1.0, 1.0)).numpy()
            if e2a_mixed["in_"].numel() > 0:
                v_out = a2ee2a_edge_vec[e2a_mixed["out"]]
                v_in = edge_vec[e2a_mixed["in_"]]
                cos = (v_out * v_in).sum(dim=-1) / (v_out.norm(dim=-1) * v_in.norm(dim=-1) + 1e-12)
                e2a_angles = torch.arccos(cos.clamp(-1.0, 1.0)).numpy()

    # ---------- Visualisation setup ----------
    vis_points = coords.copy()
    if is_periodic:
        vis_points = np.concatenate([vis_points, _cell_corner_coords(pmg_obj)], axis=0)
    if edge_vec_np.shape[0] > 0:
        vis_points = np.concatenate([vis_points, coords[edge_src] + edge_vec_np], axis=0)

    label_atoms = (not args.no_atom_labels) and (n_atoms <= 60)
    abs_sym = Element.from_Z(int(atomic_numbers[args.absorber_idx])).symbol

    # Build list of "edge graphs" to plot (always main, plus int / a2ee2a / a2a
    # when active). Each entry: (title, edge_index_np, edge_vec_np, color_intra).
    edge_panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]] = [
        (
            f"Main edges ({args.graph_method}, E={edge_index.shape[1]})",
            edge_src,
            edge_dst,
            edge_vec_np,
            (0.15, 0.45, 0.75),
        )
    ]
    if args.quadruplets and int_edge_index.size(1) > 0:
        edge_panels.append(
            (
                f"Int edges (qint, cutoff={int_cutoff}, E={int_edge_index.shape[1]})",
                int_edge_index[0].numpy(),
                int_edge_index[1].numpy(),
                int_edge_vec.numpy(),
                (0.95, 0.55, 0.10),
            )
        )
    if args.oc:
        edge_panels.append(
            (
                f"a2ee2a edges (cutoff={oc_cutoff_aeaint}, E={a2ee2a_edge_index.shape[1]})",
                a2ee2a_edge_index[0].numpy(),
                a2ee2a_edge_index[1].numpy(),
                a2ee2a_edge_vec.numpy(),
                (0.50, 0.10, 0.65),
            )
        )
        edge_panels.append(
            (
                f"a2a edges (cutoff={oc_cutoff_aint}, E={a2a_edge_index.shape[1]})",
                a2a_edge_index[0].numpy(),
                a2a_edge_index[1].numpy(),
                a2a_edge_vec.numpy(),
                (0.10, 0.55, 0.55),
            )
        )

    # Build list of "index graphs" (triplets, quads, mixed a2e/e2a).
    n_a2e = a2e_mixed.get("in_", torch.empty(0)).numel() if a2e_mixed else 0
    n_e2a = e2a_mixed.get("in_", torch.empty(0)).numel() if e2a_mixed else 0
    index_panels: list[str] = ["triplets"]
    if args.quadruplets:
        index_panels.append("quads")
    if args.oc:
        index_panels.append("a2e")
        index_panels.append("e2a")

    n_graph_cols = max(len(edge_panels), len(index_panels))
    fig = plt.figure(figsize=(5.5 * n_graph_cols + 1.0, 20.0))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=n_graph_cols,
        height_ratios=[3.2, 3.2, 1.4],
        hspace=0.32,
        wspace=0.26,
        left=0.04,
        right=0.98,
        top=0.93,
        bottom=0.05,
    )

    # ---- Row 0: edge graphs ----
    n_pbc_main = 0
    for i, (title, es, ed, evec, col) in enumerate(edge_panels):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        setup_axis(ax, pmg_obj, coords, atomic_numbers, args.absorber_idx, vis_points, is_periodic, title, label_atoms)
        if es.size == 0:
            ax.text2D(0.5, 0.5, "empty graph", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue
        if i == 0:
            n_pbc_main = plot_edges(ax, coords, es, ed, evec, is_periodic)
            leg = [Patch(color=(*col, 0.8), label="intra-cell")]
            if is_periodic:
                leg.append(Patch(color=(0.85, 0.30, 0.10, 0.85), label=f"PBC-crossing ({n_pbc_main})"))
            if args.show_voronoi:
                facets = compute_voronoi_facets(pmg_obj, args.cutoff)
                plot_voronoi_facets(ax, facets)
                leg.append(Patch(color=(0.27, 0.0, 0.33, 0.55), label=f"Voronoi facets ({len(facets)})"))
            ax.legend(handles=leg, loc="upper left", fontsize=8)
        else:
            _draw_generic_edges(ax, coords, es, ed, evec, is_periodic, color_intra=col, color_pbc=(0.85, 0.30, 0.10))
            ax.legend(handles=[Patch(color=(*col, 0.8), label=title.split(" ")[0])], loc="upper left", fontsize=8)

    # ---- Row 1: index graphs (triplets / quads / mixed) ----
    for i, kind in enumerate(index_panels):
        ax = fig.add_subplot(gs[1, i], projection="3d")
        if kind == "triplets":
            setup_axis(
                ax,
                pmg_obj,
                coords,
                atomic_numbers,
                args.absorber_idx,
                vis_points,
                is_periodic,
                f"Triplets c-a-b  (T={id3_reduce_ca.numel()}, <={args.max_triplets_drawn} drawn)",
                label_atoms,
            )
            plot_triplets(
                ax,
                coords,
                edge_src,
                edge_dst,
                edge_vec_np,
                id3_reduce_ca.numpy(),
                id3_expand_ba.numpy(),
                args.max_triplets_drawn,
            )
            ax.legend(handles=[Patch(color=(0.30, 0.75, 0.35, 0.35), label="(c, a, b)")], loc="upper left", fontsize=8)
        elif kind == "quads":
            if quad and quad["id4_reduce_ca"].numel() > 0:
                setup_axis(
                    ax,
                    pmg_obj,
                    coords,
                    atomic_numbers,
                    args.absorber_idx,
                    vis_points,
                    is_periodic,
                    f"Quadruplets c-a-b-d  (Q={quad['id4_reduce_ca'].numel()}, <={args.max_quads_drawn} drawn)",
                    label_atoms,
                )
                plot_quadruplets(
                    ax,
                    coords,
                    edge_src,
                    edge_dst,
                    edge_vec_np,
                    int_edge_index[0].numpy(),
                    int_edge_index[1].numpy(),
                    int_edge_vec.numpy(),
                    quad["id4_reduce_ca"].numpy(),
                    quad["id4_expand_db"].numpy(),
                    quad["id4_int_edge_diag"].numpy(),
                    args.max_quads_drawn,
                )
                ax.legend(
                    handles=[
                        Patch(color=(0.10, 0.55, 0.15), label="c-a (main)"),
                        Patch(color=(0.95, 0.50, 0.10), label="a-b (int)"),
                        Patch(color=(0.20, 0.40, 0.80), label="b-d (main)"),
                    ],
                    loc="upper left",
                    fontsize=8,
                )
            else:
                setup_axis(
                    ax,
                    pmg_obj,
                    coords,
                    atomic_numbers,
                    args.absorber_idx,
                    vis_points,
                    is_periodic,
                    "Quadruplets (none)",
                    label_atoms,
                )
                ax.text2D(0.5, 0.5, "no quadruplets", ha="center", va="center", transform=ax.transAxes, color="gray")
        elif kind == "a2e":
            setup_axis(
                ax,
                pmg_obj,
                coords,
                atomic_numbers,
                args.absorber_idx,
                vis_points,
                is_periodic,
                f"Mixed triplets a2e  (T={n_a2e}, <={args.max_mixed_drawn} drawn)",
                label_atoms,
            )
            if n_a2e > 0:
                plot_mixed_triplets(
                    ax,
                    coords,
                    edge_src,
                    edge_dst,
                    edge_vec_np,
                    a2ee2a_edge_index[0].numpy(),
                    a2ee2a_edge_vec.numpy(),
                    a2e_mixed["out"].numpy(),
                    a2e_mixed["in_"].numpy(),
                    args.max_mixed_drawn,
                    color=(0.85, 0.25, 0.55),
                )
            ax.legend(handles=[Patch(color=(0.85, 0.25, 0.55, 0.4), label="a2e")], loc="upper left", fontsize=8)
        elif kind == "e2a":
            setup_axis(
                ax,
                pmg_obj,
                coords,
                atomic_numbers,
                args.absorber_idx,
                vis_points,
                is_periodic,
                f"Mixed triplets e2a  (T={n_e2a}, <={args.max_mixed_drawn} drawn)",
                label_atoms,
            )
            if n_e2a > 0:
                plot_mixed_triplets(
                    ax,
                    coords,
                    a2ee2a_edge_index[0].numpy(),
                    a2ee2a_edge_index[1].numpy(),
                    a2ee2a_edge_vec.numpy(),
                    edge_index[0].numpy(),
                    edge_vec_np,
                    e2a_mixed["out"].numpy(),
                    e2a_mixed["in_"].numpy(),
                    args.max_mixed_drawn,
                    color=(0.15, 0.55, 0.55),
                    seed=1,
                )
            ax.legend(handles=[Patch(color=(0.15, 0.55, 0.55, 0.4), label="e2a")], loc="upper left", fontsize=8)

    # ---------- Histograms ----------
    # Split the bottom row into exactly 3 equal-width panels regardless of
    # how many columns the 3D grid uses.
    _third = max(1, n_graph_cols // 3)
    _rem = n_graph_cols - 2 * _third
    ax_hw = fig.add_subplot(gs[2, 0:_third])
    ax_hd = fig.add_subplot(gs[2, _third : 2 * _third])
    ax_ha = fig.add_subplot(gs[2, 2 * _third : 2 * _third + _rem])

    # Edge-weight hist (main, int, a2ee2a, a2a)
    bins = 30
    hist_series: list[tuple[np.ndarray, str, str]] = [(edge_w_np, f"main (E={edge_w_np.size})", "steelblue")]
    if int_edge_w_np.size > 0:
        hist_series.append((int_edge_w_np, f"int (E={int_edge_w_np.size})", "#d98a1d"))
    if a2ee2a_edge_w_np.size > 0:
        hist_series.append((a2ee2a_edge_w_np, f"a2ee2a (E={a2ee2a_edge_w_np.size})", "#8a3da0"))
    if a2a_edge_w_np.size > 0:
        hist_series.append((a2a_edge_w_np, f"a2a (E={a2a_edge_w_np.size})", "#1b8a7a"))
    for w, label, color in hist_series:
        ax_hw.hist(w, bins=bins, color=color, edgecolor="white", alpha=0.55, label=label)
    ax_hw.axvline(args.cutoff, color="red", ls="--", lw=1.0, label=f"main cutoff={args.cutoff}")
    if int_edge_w_np.size > 0:
        ax_hw.axvline(int_cutoff, color="#a04010", ls=":", lw=1.0, label=f"int cutoff={int_cutoff}")
    if args.oc:
        ax_hw.axvline(oc_cutoff_aint, color="#1b8a7a", ls=":", lw=1.0, label=f"a2a cutoff={oc_cutoff_aint}")
    ax_hw.set_title("edge weight distributions [A]")
    ax_hw.set_xlabel("distance")
    ax_hw.set_ylabel("count")
    ax_hw.legend(fontsize=7, loc="upper right")

    # Per-atom out-degree grouped bars (main, int, a2ee2a, a2a)
    deg_main = np.bincount(edge_src, minlength=n_atoms)
    deg_series: list[tuple[np.ndarray, str, str]] = [(deg_main, "main", "steelblue")]
    if int_edge_index.size(1) > 0:
        deg_series.append((np.bincount(int_edge_index[0].numpy(), minlength=n_atoms), "int", "#d98a1d"))
    if a2ee2a_edge_index.size(1) > 0:
        deg_series.append((np.bincount(a2ee2a_edge_index[0].numpy(), minlength=n_atoms), "a2ee2a", "#8a3da0"))
    if a2a_edge_index.size(1) > 0:
        deg_series.append((np.bincount(a2a_edge_index[0].numpy(), minlength=n_atoms), "a2a", "#1b8a7a"))
    n_series = len(deg_series)
    width = 0.8 / max(1, n_series)
    x = np.arange(n_atoms)
    for i, (d, label, color) in enumerate(deg_series):
        ax_hd.bar(x + (i - (n_series - 1) / 2) * width, d, width=width, color=color, edgecolor="white", label=label)
    ax_hd.axhline(args.max_neighbors, color="red", ls="--", lw=1.0, label=f"max_nbrs={args.max_neighbors}")
    isolated_main = np.where(deg_main == 0)[0]
    iso_txt = ""
    if isolated_main.size > 0:
        iso_txt = f"main isolated: {isolated_main.tolist()[:10]}" + ("..." if isolated_main.size > 10 else "")
        ax_hd.text(
            0.5,
            0.92,
            f"WARNING: {isolated_main.size} isolated in main\n{iso_txt}",
            transform=ax_hd.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "crimson", "alpha": 0.85, "edgecolor": "none", "pad": 3.0},
        )
    ax_hd.set_title("per-atom out-degree (all graphs)")
    ax_hd.set_xlabel("atom index")
    ax_hd.set_ylabel("# edges")
    ax_hd.legend(fontsize=7, loc="upper right")

    # Angle distributions
    any_ang = False
    if triplet_angles.size > 0:
        ax_ha.hist(
            np.rad2deg(triplet_angles),
            bins=36,
            color="seagreen",
            edgecolor="white",
            alpha=0.7,
            label=f"triplet (T={triplet_angles.size})",
        )
        any_ang = True
    if dihedrals.size > 0:
        ax_ha.hist(
            np.rad2deg(dihedrals),
            bins=36,
            color="#6a1b9a",
            edgecolor="white",
            alpha=0.55,
            label=f"quad dihedral (Q={dihedrals.size})",
        )
        any_ang = True
    if a2e_angles.size > 0:
        ax_ha.hist(
            np.rad2deg(a2e_angles),
            bins=36,
            color="#c2185b",
            edgecolor="white",
            alpha=0.5,
            label=f"a2e ({a2e_angles.size})",
        )
        any_ang = True
    if e2a_angles.size > 0:
        ax_ha.hist(
            np.rad2deg(e2a_angles),
            bins=36,
            color="#1b8a7a",
            edgecolor="white",
            alpha=0.5,
            label=f"e2a ({e2a_angles.size})",
        )
        any_ang = True
    if any_ang:
        ax_ha.set_title("angle / dihedral distributions [deg]")
        ax_ha.set_xlabel("angle")
        ax_ha.legend(fontsize=7)
    else:
        ax_ha.text(0.5, 0.5, "no angles", ha="center", va="center", transform=ax_ha.transAxes)
        ax_ha.set_axis_off()

    # ---------- Stdout summary ----------
    line = "=" * 70
    print(line)
    print(f"datasource:      {args.json_dir}")
    print(f"sample:          {stem}")
    print(f"kind:            {'periodic Structure' if is_periodic else 'Molecule'}")
    print(f"# atoms:         {n_atoms}")
    print(f"absorber:        idx={args.absorber_idx}  ({abs_sym})")
    print(f"graph method:    {args.graph_method}")
    print(f"main cutoff:     {args.cutoff} A   max_neighbors: {args.max_neighbors}")
    if args.quadruplets:
        print(f"int  cutoff:     {int_cutoff} A")
    if args.oc:
        print(f"oc_cutoff_aeaint:{oc_cutoff_aeaint} A   max_nbrs: {oc_max_nbrs_aeaint}")
        print(f"oc_cutoff_aint:  {oc_cutoff_aint} A   max_nbrs: {oc_max_nbrs_aint}")
    print(line)

    print(
        f"MAIN       E={edge_index.shape[1]:6d}  PBC-cross={n_pbc_main}"
        f"  deg[min/mean/max]={int(deg_main.min())}/{deg_main.mean():.2f}/{int(deg_main.max())}"
        f"  saturated={int((deg_main >= args.max_neighbors).sum())}/{n_atoms}"
    )
    if edge_w_np.size:
        print(f"  weight[min/mean/max] = {edge_w_np.min():.3f}/{edge_w_np.mean():.3f}/{edge_w_np.max():.3f}")
    if edge_attr is not None and edge_attr.numel() > 0:
        ea = edge_attr.numpy()
        print(f"  facet area [min/mean/max] = {ea.min():.3g}/{ea.mean():.3g}/{ea.max():.3g}")
    print(f"  id_swap involution: {'OK' if id_swap_ok else f'FAIL ({id_swap_err})'}")
    if isolated_main.size > 0:
        print(f"  !! {isolated_main.size} isolated atom(s): {isolated_main.tolist()}")

    print(f"TRIPLETS   T={id3_reduce_ca.numel():6d}")
    if triplet_angles.size:
        a = np.rad2deg(triplet_angles)
        n_zero = int((a < 1e-3).sum())
        print(f"  angle[deg, min/mean/max]: {a.min():.1f}/{a.mean():.1f}/{a.max():.1f}" f"  near-zero: {n_zero}")

    if args.quadruplets:
        n_int = int_edge_index.size(1)
        deg_int = np.bincount(int_edge_index[0].numpy(), minlength=n_atoms) if n_int else np.zeros(n_atoms, dtype=int)
        print(
            f"INT EDGES  E={n_int:6d}  deg[min/mean/max]={int(deg_int.min())}/{deg_int.mean():.2f}/{int(deg_int.max())}"
        )
        n_quads = quad.get("id4_reduce_ca", torch.empty(0)).numel()
        print(
            f"QUADS      Q={n_quads:6d}"
            f"  intm_ca={quad.get('id4_reduce_intm_ca', torch.empty(0)).numel()}"
            f"  intm_db={quad.get('id4_expand_intm_db', torch.empty(0)).numel()}"
        )
        if dihedrals.size:
            d = np.rad2deg(dihedrals)
            print(f"  dihedral[deg, min/mean/max]: {d.min():.1f}/{d.mean():.1f}/{d.max():.1f}")
            n_missing = int((quad["id4_int_edge_diag"] < 0).sum().item()) if "id4_int_edge_diag" in quad else 0
            if n_missing > 0:
                print(f"  ! {n_missing} quadruplets without a matching int edge for dihedral (image-specific)")

    if args.oc:
        deg_aea = (
            np.bincount(a2ee2a_edge_index[0].numpy(), minlength=n_atoms)
            if a2ee2a_edge_index.size(1)
            else np.zeros(n_atoms, dtype=int)
        )
        deg_aa = (
            np.bincount(a2a_edge_index[0].numpy(), minlength=n_atoms)
            if a2a_edge_index.size(1)
            else np.zeros(n_atoms, dtype=int)
        )
        print(
            f"A2EE2A     E={a2ee2a_edge_index.shape[1]:6d}"
            f"  deg[min/mean/max]={int(deg_aea.min())}/{deg_aea.mean():.2f}/{int(deg_aea.max())}"
        )
        print(
            f"A2A        E={a2a_edge_index.shape[1]:6d}"
            f"  deg[min/mean/max]={int(deg_aa.min())}/{deg_aa.mean():.2f}/{int(deg_aa.max())}"
        )
        n_a2e = a2e_mixed.get("in_", torch.empty(0)).numel() if a2e_mixed else 0
        n_e2a = e2a_mixed.get("in_", torch.empty(0)).numel() if e2a_mixed else 0
        print(f"MIXED TRIP a2e={n_a2e}   e2a={n_e2a}")
        if a2e_angles.size:
            a = np.rad2deg(a2e_angles)
            print(f"  a2e angle[deg, min/mean/max]: {a.min():.1f}/{a.mean():.1f}/{a.max():.1f}")
        if e2a_angles.size:
            a = np.rad2deg(e2a_angles)
            print(f"  e2a angle[deg, min/mean/max]: {a.min():.1f}/{a.mean():.1f}/{a.max():.1f}")

    print(line)

    # ---------- Input-size impact ----------
    # Per-sample tensor-element counts that flow into the model. These dominate
    # runtime/memory scaling: edge-indexed features scale with E, triplet-
    # indexed with T, quadruplet-indexed with Q, and basis-embedding scatters
    # scale with the corresponding dst-atom neighbor lists.
    print("Input-size impact (per-sample tensor counts):")
    print(f"  atoms                              N = {n_atoms}")
    print(f"  main edges                         E_main   = {edge_index.shape[1]}")
    print(f"  triplets (c-a-b)                   T_main   = {id3_reduce_ca.numel()}")
    if args.quadruplets:
        n_int = int_edge_index.size(1)
        n_quad = quad.get("id4_reduce_ca", torch.empty(0)).numel()
        n_intm_ab = quad.get("id4_reduce_intm_ab", torch.empty(0)).numel()
        print(f"  int edges (qint)                   E_int    = {n_int}")
        print(f"  intm quadruplet-triplets (a-b)     T_intm   = {n_intm_ab}")
        print(f"  quadruplets (c-a-b-d)              Q        = {n_quad}")
        ratio_qe = (n_quad / n_int) if n_int else 0.0
        ratio_qt = (n_quad / id3_reduce_ca.numel()) if id3_reduce_ca.numel() else 0.0
        print(f"  ratios                             Q/E_int={ratio_qe:.2f}   Q/T_main={ratio_qt:.2f}")
    if args.oc:
        n_aea = a2ee2a_edge_index.shape[1]
        n_aa = a2a_edge_index.shape[1]
        print(f"  a2ee2a edges                       E_aea    = {n_aea}")
        print(f"  a2a edges                          E_aa     = {n_aa}")
        print(f"  mixed triplets atom->edge          T_a2e    = {n_a2e}")
        print(f"  mixed triplets edge->atom          T_e2a    = {n_e2a}")
        total_edges = edge_index.shape[1] + int_edge_index.size(1) + n_aea + n_aa
        total_triplets = id3_reduce_ca.numel() + n_a2e + n_e2a
        total_quads = quad.get("id4_reduce_ca", torch.empty(0)).numel() if args.quadruplets else 0
        print(f"  TOTAL edges across all graphs      sum(E)   = {total_edges}")
        print(f"  TOTAL triplet-like indices         sum(T)   = {total_triplets}")
        if args.quadruplets:
            print(f"  TOTAL quadruplet-like indices      sum(Q)   = {total_quads}")
    print(line)

    fig.suptitle(
        f"{'periodic Structure' if is_periodic else 'Molecule'}  .  {stem}  "
        f".  absorber={abs_sym}{args.absorber_idx}  .  method={args.graph_method}  "
        f".  cutoff={args.cutoff}"
        + (f"  .  int={int_cutoff}" if args.quadruplets else "")
        + (f"  .  OC(aea={oc_cutoff_aeaint}, aa={oc_cutoff_aint})" if args.oc else ""),
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
