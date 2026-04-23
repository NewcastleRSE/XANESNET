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

import torch
from torch_geometric.typing import SparseTensor


def _ragged_range(sizes: torch.Tensor) -> torch.Tensor:
    """
    Multiple concatenated ranges.

    Example: ``sizes = [1, 4, 2, 3]`` -> ``[0, 0,1,2,3, 0,1, 0,1,2]``.
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
    """
    Round displacement vectors to tolerate tiny numerical differences.
    """
    scale = 10**decimals
    return (vec * scale).round().to(torch.int64)


def compute_id_swap(edge_index: torch.Tensor, edge_vec: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    """
    For each directed edge ``e = (c -> a)`` with vector ``v``, return the index of its counter-edge
    ``(a -> c)`` with vector ``-v``. Works for molecular and periodic graphs provided the graph is
    already symmetrised (every forward edge has a reverse partner, possibly at the same image).

    Raises
    ------
    ValueError
        If no matching counter-edge is found for some edge.
    """
    e = edge_index.size(1)
    if e == 0:
        return edge_index.new_empty(0, dtype=torch.int64)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    v_r = _round_vec(edge_vec, decimals).tolist()

    fwd: dict[tuple, int] = {}
    for i in range(e):
        key = (src[i], dst[i], v_r[i][0], v_r[i][1], v_r[i][2])
        fwd[key] = i

    id_swap = torch.empty(e, dtype=torch.int64, device=edge_index.device)
    for i in range(e):
        rev_key = (dst[i], src[i], -v_r[i][0], -v_r[i][1], -v_r[i][2])
        j = fwd.get(rev_key)
        if j is None:
            raise ValueError(
                f"Edge {i} (src={src[i]}, dst={dst[i]}) has no matching counter-edge; graph is not symmetric."
            )
        id_swap[i] = j
    return id_swap


def compute_triplets(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute GemNet triplet indices ``c -> a <- b`` where both edges target the same atom ``a``.

    Returns
    -------
    id3_reduce_ca: (T,) int64 edge indices, giving the ``c -> a`` edge of each
        triplet. Sorted ascending so ``Kidx3`` is well-defined.
    id3_expand_ba: (T,) int64 edge indices, giving the ``b -> a`` edge of each
        triplet.
    Kidx3: (T,) ragged inner index within each ``id3_reduce_ca`` group
        (``[0, 1, ..., K_i - 1]``).

    Notes
    -----
    Self-loop triplets where ``e == e'`` (the same directed edge used as both legs) are removed.
    Two edges with the same atom indices but different periodic images give a valid triplet (they have different edge ids).
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
    """
    Compute GemNet-Q / GemNet-OC quadruplet indices for quadruplets
    ``c -> a - b <- d`` where

    - ``(c -> a)`` and ``(d -> b)`` are edges of the **main** graph (embedding
      cutoff), and
    - ``(b -> a)`` is an edge of the **interaction** graph (``int_cutoff``,
      typically larger).

    Degenerate quadruplets (``c == b``, ``a == d``, ``c == d`` in the same periodic image) are filtered using the path-vector test:

        vec_cb = vec_ca - vec_ba       (pos_b - pos_c)
        vec_ad = -(vec_ba + vec_db)    (pos_d - pos_a)
        vec_cd = vec_ca - vec_ba - vec_db  (pos_d - pos_c)

    For each identity test we also require the atom indices match; otherwise a near-zero vector
    from two genuinely different atoms would incorrectly be filtered.

    Returns
    -------
    dict with the standard GemNet index tensors
    ``{id4_reduce_ca, id4_expand_db, id4_reduce_cab, id4_expand_abd,
       id4_reduce_intm_ca, id4_expand_intm_db, id4_reduce_intm_ab,
       id4_expand_intm_ab, Kidx4}``.
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
    """
    Mixed triplet indices used by GemNet-OC's atom-edge and edge-atom
    interactions. For each "output" edge ``(c -> a)`` in ``main_edge_index``,
    enumerate all "input" edges in ``other_edge_index`` that connect to the
    same atom (either ``a`` or ``c``, depending on ``to_outedge``).

    Parameters
    ----------
    to_outedge:
        If False (the "ingoing" case, used for atom-edge / edge-atom in OC),
        match input edges to the target atom ``a`` of the output edge.
        If True (the GemNet-OC quad "triplet_in" case), match input edges to
        the source atom ``c`` of the output edge.

    Returns
    -------
    dict with ``in`` (input edge ids), ``out`` (output edge ids),
    ``out_agg`` (ragged inner index enumerating inputs per output).

    Degenerate self-loop mixed triplets are removed using the path-vector
    test (same atom AND path vector ~ 0 implies same periodic image).
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
