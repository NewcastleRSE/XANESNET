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

"""GemNet triplet, quadruplet, and mixed-triplet index computation."""

import torch
from torch_geometric.typing import SparseTensor


def _edge_keys(edge_index: torch.Tensor, edge_vec: torch.Tensor, decimals: int) -> list[tuple[int, int, int, int, int]]:
    """Return quantized per-edge keys for reverse-edge matching.

    Args:
        edge_index: ``(2, E)`` int64 source/destination indices.
        edge_vec: ``(E, 3)`` float displacement vectors.
        decimals: Number of decimal places used before integer quantization.

    Returns:
        List of per-edge keys ``(src, dst, qx, qy, qz)``.
    """
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    v_r = _round_vec(edge_vec, decimals).tolist()
    return [(src_i, dst_i, vec_i[0], vec_i[1], vec_i[2]) for src_i, dst_i, vec_i in zip(src, dst, v_r, strict=True)]


def _reverse_key(key: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    """Return the reverse-edge key corresponding to ``key``.

    Args:
        key: Quantized edge key ``(src, dst, qx, qy, qz)``.

    Returns:
        Reverse-edge key ``(dst, src, -qx, -qy, -qz)``.
    """
    src, dst, qx, qy, qz = key
    return dst, src, -qx, -qy, -qz


def _ragged_range(sizes: torch.Tensor) -> torch.Tensor:
    """Return concatenated ranges ``[0, 1, ..., s_i - 1]`` for each size ``s_i``.

    Example:
        ``sizes = [1, 4, 2, 3]`` -> ``[0, 0, 1, 2, 3, 0, 1, 0, 1, 2]``

    Args:
        sizes: ``(N,)`` tensor of non-negative range lengths.

    Returns:
        ``(sum(sizes),)`` ragged-range tensor.
    """
    assert sizes.dim() == 1
    if int(sizes.sum().item()) == 0:
        return sizes.new_empty(0)

    nz = sizes > 0
    if not bool(nz.all()):
        sizes = sizes[nz]

    id_steps = torch.ones(int(sizes.sum().item()), dtype=torch.long, device=sizes.device)
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(0)
    insert_val = (1 - sizes)[:-1]
    id_steps[insert_index] = insert_val
    return id_steps.cumsum(0)


def _round_vec(vec: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    """Round displacement vectors to integer-quantised values for key comparison.

    Args:
        vec: ``(E, 3)`` float displacement vectors.
        decimals: Number of decimal places to round to before quantising.

    Returns:
        ``(E, 3)`` int64 tensor.
    """
    scale = 10**decimals
    return (vec * scale).round().to(torch.int64)


def compute_id_swap(edge_index: torch.Tensor, edge_vec: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    """For each directed edge ``(c -> a, v)`` return the index of its counter-edge ``(a -> c, -v)``.

    Works for molecular and periodic graphs provided the graph is already
    symmetrised (every forward edge has a reverse partner, possibly at the
    same image).

    Args:
        edge_index: ``(2, E)`` int64 -- source/destination node indices.
        edge_vec: ``(E, 3)`` float -- displacement vectors.
        decimals: Decimal places used when quantising vectors for matching.

    Returns:
        ``(E,)`` int64 -- for each edge ``i``, the index of its reverse edge.

    Raises:
        ValueError: If no matching counter-edge is found for any edge.
    """
    e = edge_index.size(1)
    if e == 0:
        return edge_index.new_empty(0, dtype=torch.int64)

    keys = _edge_keys(edge_index, edge_vec, decimals)
    buckets: dict[tuple[int, int, int, int, int], list[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault(key, []).append(i)

    id_swap = torch.empty(e, dtype=torch.int64, device=edge_index.device)
    processed: set[tuple[int, int, int, int, int]] = set()
    for key, idxs in buckets.items():
        if key in processed:
            continue

        rev_key = _reverse_key(key)
        rev_idxs = buckets.get(rev_key)
        if rev_idxs is None:
            i = idxs[0]
            src_i, dst_i, *_ = key
            raise ValueError(
                f"Edge {i} (src={src_i}, dst={dst_i}) has no matching counter-edge; graph is not symmetric."
            )

        if key == rev_key:
            for i in idxs:
                id_swap[i] = i
            processed.add(key)
            continue

        if len(idxs) != len(rev_idxs):
            src_i, dst_i, *_ = key
            raise ValueError(
                "Edge multiplicities do not match between forward and reverse directions "
                f"for (src={src_i}, dst={dst_i})."
            )

        for i, j in zip(idxs, rev_idxs, strict=True):
            id_swap[i] = j
            id_swap[j] = i

        processed.add(key)
        processed.add(rev_key)

    return id_swap


def compute_triplets(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GemNet triplet indices ``c -> a <- b`` sharing target atom ``a``.

    Args:
        edge_index: ``(2, E)`` int64 -- directed edge list.
        num_nodes: Total number of nodes in the graph.

    Returns:
        A 3-tuple ``(id3_reduce_ca, id3_expand_ba, Kidx3)``:

        - ``id3_reduce_ca``: ``(T,)`` int64 -- edge indices of the ``c -> a``
          leg, sorted ascending so ``Kidx3`` is well-defined.
        - ``id3_expand_ba``: ``(T,)`` int64 -- edge indices of the ``b -> a``
          leg.
        - ``Kidx3``: ``(T,)`` int64 -- ragged inner index within each
          ``id3_reduce_ca`` group (``[0, 1, ..., K_i - 1]``).

    Note:
        Self-loop triplets where the same directed edge serves as both legs
        are removed. Two edges with the same atom indices but different
        periodic images yield valid, distinct triplets.
    """
    n_edges = edge_index.size(1)
    if n_edges == 0:
        empty = edge_index.new_empty(0, dtype=torch.int64)
        return empty, empty, empty

    idx_s = edge_index[0]
    idx_t = edge_index[1]
    value = torch.arange(n_edges, device=edge_index.device, dtype=idx_s.dtype)

    # adj[a, c] stores edge id of (c -> a). Select rows idx_t (target) -> all edges sharing the same target atom a.
    adj = SparseTensor(
        row=idx_t,
        col=idx_s,
        value=value,
        sparse_sizes=(num_nodes, num_nodes),
    )
    adj_sel = adj.index_select(0, idx_t)  # type: ignore[attr-defined]  # rows selected in order of edges

    id3_expand_ba = adj_sel.storage.value()  # edges into a
    id3_reduce_ca = adj_sel.storage.row()  # which output edge (= e, the c->a edge)

    # Remove e == e' (same edge used twice). Different periodic images of the
    # same atom pair have distinct edge ids, so they survive here.
    mask = id3_reduce_ca != id3_expand_ba
    id3_reduce_ca = id3_reduce_ca[mask]
    id3_expand_ba = id3_expand_ba[mask]

    # Sort by id3_reduce_ca so Kidx3 is contiguous per output edge.
    sorted_idx = torch.argsort(id3_reduce_ca, stable=True)
    id3_reduce_ca = id3_reduce_ca[sorted_idx]
    id3_expand_ba = id3_expand_ba[sorted_idx]

    # Ragged inner index: for each unique id3_reduce_ca value, enumerate 0..K-1.
    counts = torch.zeros(n_edges, dtype=torch.int64, device=edge_index.device)
    ones = torch.ones_like(id3_reduce_ca)
    counts.scatter_add_(0, id3_reduce_ca, ones)
    Kidx3 = _ragged_range(counts[counts > 0]) if counts.sum() > 0 else counts.new_empty(0)
    # Note: counts[counts>0] preserves order of first-occurrence of groups;
    # since id3_reduce_ca is sorted, this matches the groups in order.

    return id3_reduce_ca.to(torch.int64), id3_expand_ba.to(torch.int64), Kidx3.to(torch.int64)


def compute_quadruplets(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    int_edge_index: torch.Tensor,
    int_edge_vec: torch.Tensor,
    num_nodes: int,
    eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Compute GemNet-Q / GemNet-OC quadruplet indices.

    Quadruplets have the form ``c -> a - b <- d`` where:

    - ``(c -> a)`` and ``(d -> b)`` are edges of the **main** graph (embedding
      cutoff), and
    - ``(b -> a)`` is an edge of the **interaction** graph (``int_cutoff``,
      typically larger).

    Degenerate quadruplets (``c == b``, ``a == d``, ``c == d`` at the same
    periodic image) are filtered using path-vector tests:

    - ``vec_cb = vec_ca - vec_ba``  (pos_b - pos_c)
    - ``vec_ad = -(vec_ba + vec_db)``  (pos_d - pos_a)
    - ``vec_cd = vec_ca - vec_ba - vec_db``  (pos_d - pos_c)

    For each identity test, the atom indices must also match; otherwise a
    near-zero vector from two genuinely different atoms would be incorrectly
    filtered.

    Args:
        edge_index: ``(2, E)`` int64 -- main-graph directed edges.
        edge_vec: ``(E, 3)`` float -- main-graph displacement vectors.
        int_edge_index: ``(2, E_int)`` int64 -- interaction-graph edges.
        int_edge_vec: ``(E_int, 3)`` float -- interaction displacement vectors.
        num_nodes: Total number of nodes.
        eps: Distance threshold below which two positions are considered
            identical (used for degeneracy filtering).

    Returns:
        Dictionary with the standard GemNet index tensors:
        ``id4_reduce_ca``, ``id4_expand_db``, ``id4_reduce_cab``,
        ``id4_expand_abd``, ``id4_reduce_intm_ca``, ``id4_expand_intm_db``,
        ``id4_reduce_intm_ab``, ``id4_expand_intm_ab``, ``Kidx4``.
    """
    device = edge_index.device
    n_edges = edge_index.size(1)
    n_int = int_edge_index.size(1)

    # Interaction edge b -> a: source=b, target=a
    idx_int_s = int_edge_index[0]  # b
    idx_int_t = int_edge_index[1]  # a

    # Main-graph edge c -> a: source=c, target=a
    idx_s = edge_index[0]
    idx_t = edge_index[1]

    # Build sparse adjacency of main graph edges indexed by target atom:
    # adj[a, c] = edge id of (c -> a).
    value = torch.arange(n_edges, device=device, dtype=torch.int64)
    adj = SparseTensor(
        row=idx_t,
        col=idx_s,
        value=value,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # For each interaction edge b -> a:
    #   intermediate "c -> a" edges: all main-graph edges ending at a (row=a)
    #   intermediate "d -> b" edges: all main-graph edges ending at b (row=b)
    adj_ca = adj.index_select(0, idx_int_t)  # type: ignore[attr-defined]
    adj_db = adj.index_select(0, idx_int_s)  # type: ignore[attr-defined]

    id4_reduce_intm_ca = adj_ca.storage.value().to(torch.int64)  # main edge ids (c->a)
    id4_expand_intm_db = adj_db.storage.value().to(torch.int64)  # main edge ids (d->b)

    # Number of c->a per interaction edge (grouped by target a)
    n_ca_per_int = adj_ca.storage.row().bincount(minlength=n_int).to(torch.int64)
    n_db_per_int = adj_db.storage.row().bincount(minlength=n_int).to(torch.int64)

    # Intermediate "ab" index (maps each intermediate edge to its int edge)
    id4_reduce_intm_ab = torch.repeat_interleave(torch.arange(n_int, device=device, dtype=torch.int64), n_ca_per_int)
    id4_expand_intm_ab = torch.repeat_interleave(torch.arange(n_int, device=device, dtype=torch.int64), n_db_per_int)

    # -------- Build full cartesian quadruplets: for each int edge, pair its
    # (c,a) edges with its (d,b) edges. --------
    # Inside int edge i there are n_ca*n_db quadruplets.
    total_quads = int((n_ca_per_int * n_db_per_int).sum().item())

    if total_quads == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return dict(
            id4_reduce_ca=empty,
            id4_expand_db=empty,
            id4_reduce_cab=empty,
            id4_expand_abd=empty,
            id4_reduce_intm_ca=id4_reduce_intm_ca,
            id4_expand_intm_db=id4_expand_intm_db,
            id4_reduce_intm_ab=id4_reduce_intm_ab,
            id4_expand_intm_ab=id4_expand_intm_ab,
            Kidx4=empty,
        )

    # Per-int-edge cumulative offsets into intermediate arrays.
    ca_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            n_ca_per_int.cumsum(0)[:-1],
        ]
    )
    db_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            n_db_per_int.cumsum(0)[:-1],
        ]
    )

    # For each int edge i, produce id4_reduce_cab (local ca index -> intm ca idx)
    # as the cartesian product "n_db[i] repeats of range(n_ca[i]) + ca_offset[i]".
    # id4_expand_abd: "range(n_db[i]) repeated n_ca[i] times + db_offset[i]".
    # Fully vectorised: per-int-edge block has size n_ca[i]*n_db[i]; within
    # each block local_j in [0, sizes[i]) encodes (db = j // nca, ca = j % nca).
    sizes = n_ca_per_int * n_db_per_int
    quad_int_edge = torch.repeat_interleave(torch.arange(n_int, device=device, dtype=torch.int64), sizes)
    local_j = _ragged_range(sizes)
    nca_per_quad = n_ca_per_int[quad_int_edge]
    ca_base_per_quad = ca_offsets[quad_int_edge]
    db_base_per_quad = db_offsets[quad_int_edge]
    id4_reduce_cab = ca_base_per_quad + (local_j % nca_per_quad)
    id4_expand_abd = db_base_per_quad + (local_j // nca_per_quad)

    id4_reduce_ca = id4_reduce_intm_ca[id4_reduce_cab]
    id4_expand_db = id4_expand_intm_db[id4_expand_abd]

    # --- Degeneracy filtering (edge_vec based) ---
    # edge_vec direction: main edge_vec[e] = pos_target - pos_source
    #   edge c->a: vec_ca = pos_a - pos_c
    #   edge d->b: vec_db = pos_b - pos_d
    # interaction edge b->a: int_edge_vec[e] = pos_a - pos_b = vec_ba
    vec_ca = edge_vec[id4_reduce_ca]
    vec_db = edge_vec[id4_expand_db]
    vec_ba = int_edge_vec[quad_int_edge]

    # Atom indices
    idx_c = idx_s[id4_reduce_ca]
    idx_a = idx_t[id4_reduce_ca]
    # d->b means src=d, dst=b. So idx_t[db] = b, idx_s[db] = d.
    idx_b = idx_t[id4_expand_db]
    idx_d = idx_s[id4_expand_db]

    # vec_cb = pos_b - pos_c = vec_ca - vec_ba
    vec_cb = vec_ca - vec_ba
    # vec_ad = pos_d - pos_a = -(vec_ba + vec_db)
    vec_ad = -(vec_ba + vec_db)
    # vec_cd = pos_d - pos_c = vec_ca - vec_ba - vec_db
    vec_cd = vec_ca - vec_ba - vec_db

    mask_cb = (idx_c != idx_b) | (vec_cb.norm(dim=-1) > eps)
    mask_ad = (idx_a != idx_d) | (vec_ad.norm(dim=-1) > eps)
    mask_cd = (idx_c != idx_d) | (vec_cd.norm(dim=-1) > eps)
    mask = mask_cb & mask_ad & mask_cd

    id4_reduce_ca = id4_reduce_ca[mask]
    id4_expand_db = id4_expand_db[mask]
    id4_reduce_cab = id4_reduce_cab[mask]
    id4_expand_abd = id4_expand_abd[mask]

    if id4_reduce_ca.numel() == 0:
        Kidx4 = torch.empty(0, dtype=torch.int64, device=device)
    else:
        sorted_idx = torch.argsort(id4_reduce_ca, stable=True)
        id4_reduce_ca = id4_reduce_ca[sorted_idx]
        id4_expand_db = id4_expand_db[sorted_idx]
        id4_reduce_cab = id4_reduce_cab[sorted_idx]
        id4_expand_abd = id4_expand_abd[sorted_idx]

        counts = torch.zeros(n_edges, dtype=torch.int64, device=device)
        ones = torch.ones_like(id4_reduce_ca)
        counts.scatter_add_(0, id4_reduce_ca, ones)
        Kidx4 = _ragged_range(counts[counts > 0])

    return dict(
        id4_reduce_ca=id4_reduce_ca.to(torch.int64),
        id4_expand_db=id4_expand_db.to(torch.int64),
        id4_reduce_cab=id4_reduce_cab.to(torch.int64),
        id4_expand_abd=id4_expand_abd.to(torch.int64),
        id4_reduce_intm_ca=id4_reduce_intm_ca,
        id4_expand_intm_db=id4_expand_intm_db,
        id4_reduce_intm_ab=id4_reduce_intm_ab,
        id4_expand_intm_ab=id4_expand_intm_ab,
        Kidx4=Kidx4.to(torch.int64),
    )


def compute_mixed_triplets(
    main_edge_index: torch.Tensor,
    main_edge_vec: torch.Tensor,
    other_edge_index: torch.Tensor,
    other_edge_vec: torch.Tensor,
    num_nodes: int,
    to_outedge: bool,
    eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Compute mixed-triplet indices for GemNet-OC atom-edge / edge-atom interactions.

    For each "output" edge ``(c -> a)`` in ``main_edge_index``, enumerates all
    "input" edges in ``other_edge_index`` that connect to the same atom (either
    ``a`` or ``c``, depending on ``to_outedge``).

    Degenerate self-loop mixed triplets are removed via path-vector tests
    (same atom AND path vector near zero implies same periodic image).

    Args:
        main_edge_index: ``(2, E_out)`` int64 -- output directed edge list.
        main_edge_vec: ``(E_out, 3)`` float -- output displacement vectors.
        other_edge_index: ``(2, E_in)`` int64 -- input directed edge list.
        other_edge_vec: ``(E_in, 3)`` float -- input displacement vectors.
        num_nodes: Total number of nodes.
        to_outedge: If ``False`` (atom-edge / edge-atom case), match input
            edges to the target atom ``a`` of each output edge. If ``True``
            (GemNet-OC ``triplet_in`` case), match input edges to the source
            atom ``c`` of each output edge.
        eps: Distance threshold for degeneracy filtering.

    Returns:
        Dictionary with keys ``in_`` (input edge ids), ``out`` (output edge
        ids), and ``out_agg`` (ragged inner index enumerating inputs per
        output edge).
    """
    device = main_edge_index.device
    n_out = main_edge_index.size(1)
    if n_out == 0 or other_edge_index.size(1) == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return dict(in_=empty, out=empty, out_agg=empty)

    idx_out_s = main_edge_index[0]
    idx_out_t = main_edge_index[1]
    idx_in_s = other_edge_index[0]
    idx_in_t = other_edge_index[1]

    value_in = torch.arange(other_edge_index.size(1), device=device, dtype=torch.int64)
    # For input graph: adj[target, source] = edge_id
    adj_in = SparseTensor(
        row=idx_in_t,
        col=idx_in_s,
        value=value_in,
        sparse_sizes=(num_nodes, num_nodes),
    )

    pivot = idx_out_s if to_outedge else idx_out_t
    adj_sel = adj_in.index_select(0, pivot)  # type: ignore[attr-defined]
    idx_in = adj_sel.storage.value().to(torch.int64)
    idx_out = adj_sel.storage.row().to(torch.int64)

    # Degeneracy: remove c->a<-c / c<-a<-c self loops where the shared atom is
    # at the same periodic image.
    if to_outedge:
        # Output edge (a->c in this framing): check in-source atom vs out-target
        # Actually in GemNet-OC get_mixed_triplets with to_outedge=True, they
        # use: idx_atom_in = idx_in_s[idx_in]; idx_atom_out = idx_out_t[idx_out]
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_t[idx_out]
        # Path vector: out edge is (c -> a) with vec_ca = pos_a - pos_c pivot=idx_out_s=c
        # Pivot shared atom is at pos_c for output, pos_in_t for input.
        # Input edge is (p -> c) with vec_pc = pos_c - pos_p; source p = idx_in_s[idx_in]
        # We want to test if (target of input == target of output in absolute
        # space) i.e., p==a at same image. Path: start at pos_c (pivot) -> go
        # along input's reverse to source p: pos_p = pos_c - vec_pc; then need
        # to check pos_p == pos_a: pos_c - vec_pc == pos_c + vec_ca -> vec_pc + vec_ca == 0.
        # Actually gemnet_oc uses cell_offsets_sum; for us we use:
        # diff = vec_out_source_to_target + vec_in_source_to_target (ways to reach same "other end").
        # Conservative simple test: same atoms AND same path-vector sum.
        v_out = main_edge_vec[idx_out]  # pos_a - pos_c
        v_in = other_edge_vec[idx_in]  # pos_t_in - pos_s_in
        path = v_out + v_in
        mask = (idx_atom_in != idx_atom_out) | (path.norm(dim=-1) > eps)
    else:
        # Pivot shared atom is the target a. Source of output is c; source of
        # input is b (with target a). Degenerate if c == b at same image,
        # i.e., vec_out(c->a) == vec_in(b->a) (both end at same pos_a).
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_s[idx_out]
        v_out = main_edge_vec[idx_out]  # pos_a - pos_c
        v_in = other_edge_vec[idx_in]  # pos_a - pos_b
        diff = v_out - v_in
        mask = (idx_atom_in != idx_atom_out) | (diff.norm(dim=-1) > eps)

    idx_in = idx_in[mask]
    idx_out = idx_out[mask]

    # Sort by out for ragged out_agg
    sorted_idx = torch.argsort(idx_out, stable=True)
    idx_out = idx_out[sorted_idx]
    idx_in = idx_in[sorted_idx]

    counts = torch.zeros(n_out, dtype=torch.int64, device=device)
    ones = torch.ones_like(idx_out)
    counts.scatter_add_(0, idx_out, ones)
    out_agg = _ragged_range(counts[counts > 0]) if counts.sum() > 0 else counts.new_empty(0)

    return dict(in_=idx_in, out=idx_out, out_agg=out_agg)
