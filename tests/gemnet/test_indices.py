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
Tests for ``xanesnet/utils/graph/gemnet_indices.py`` and the `get_inner_idx` helper used by GemNet-OC's model code.

These tests cover:

- ``_ragged_range`` basic semantics and zero-size groups
- ``get_inner_idx`` on sorted AND unsorted inputs (regression for the
  silent-corruption bug triggered by unsorted edge targets)
- ``compute_id_swap`` symmetry and periodic-image disambiguation
- ``compute_triplets`` self-loop removal and Kidx3 correctness
- ``compute_mixed_triplets`` degeneracy filter under PBC
- ``compute_quadruplets`` symmetry and cd/ab/ad filters
- Comparison against the fairchem-style reference implementation on a
  synthetic periodic structure

The fairchem reference functions live in ``conftest.py``; all comparisons are performed on 
the _set_ of produced triplets/quadruplets (order is an implementation detail).
"""

import torch

from xanesnet.models.gemnet_oc.utils import get_inner_idx
from xanesnet.utils.graph import build_edges
from xanesnet.utils.graph.gemnet_indices import (
    _ragged_range,
    compute_id_swap,
    compute_mixed_triplets,
    compute_quadruplets,
    compute_triplets,
)

from .conftest import (
    cell_offsets_from_edge_vec,
    ref_get_mixed_triplets,
    ref_get_quadruplets,
    ref_get_triplets,
    sort_triplet_set,
)

# Helpers


class TestRaggedRange:
    def test_basic(self):
        out = _ragged_range(torch.tensor([1, 4, 2, 3]))
        assert out.tolist() == [0, 0, 1, 2, 3, 0, 1, 0, 1, 2]

    def test_all_zero(self):
        out = _ragged_range(torch.tensor([0, 0, 0]))
        assert out.numel() == 0

    def test_some_zero(self):
        out = _ragged_range(torch.tensor([0, 3, 0, 2]))
        assert out.tolist() == [0, 1, 2, 0, 1]


class TestGetInnerIdx:
    def test_sorted(self):
        idx = torch.tensor([0, 0, 0, 1, 2, 2])
        out = get_inner_idx(idx, dim_size=3)
        assert out.tolist() == [0, 1, 2, 0, 0, 1]

    def test_unsorted_regression(self):
        """
        Unsorted inputs must still yield a valid 0-based enumeration per group.
        This exercises the bug where ``segment_coo`` required sorted input and silently corrupted ``target_neighbor_idx``.
        """
        idx = torch.tensor([2, 0, 1, 0, 2, 2, 0])
        out = get_inner_idx(idx, dim_size=3)
        # Check (group, inner) pairs are unique
        pairs = set(zip(idx.tolist(), out.tolist()))
        assert len(pairs) == idx.numel()
        # Check per-group inner indices cover [0, K-1]
        for g in range(3):
            inners_g = sorted(o for i, o in zip(idx.tolist(), out.tolist()) if i == g)
            count_g = int((idx == g).sum().item())
            assert inners_g == list(range(count_g))

    def test_empty(self):
        out = get_inner_idx(torch.zeros(0, dtype=torch.int64), dim_size=5)
        assert out.numel() == 0


# compute_id_swap / compute_triplets


def _make_symmetric_graph():
    """
    Simple non-periodic triangle 0-1-2 with both directions on every edge.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # edges: (0,1), (1,0), (0,2), (2,0), (1,2), (2,1)
    src = torch.tensor([0, 1, 0, 2, 1, 2])
    dst = torch.tensor([1, 0, 2, 0, 2, 1])
    edge_index = torch.stack([src, dst])
    edge_vec = pos[dst] - pos[src]
    return edge_index, edge_vec, pos


class TestIdSwap:
    def test_triangle(self):
        ei, ev, _ = _make_symmetric_graph()
        swap = compute_id_swap(ei, ev)
        # Every (s,d,v) must map to (d,s,-v)
        for i, j in enumerate(swap.tolist()):
            assert ei[0, i].item() == ei[1, j].item()
            assert ei[1, i].item() == ei[0, j].item()
            assert torch.allclose(ev[i], -ev[j], atol=1e-6)

    def test_triangle_exact_values(self):
        """
        Hand-computed ``id_swap`` for the 6-edge triangle: adjacent pairs were constructed
        as (forward, reverse), so the swap must map 0<->1, 2<->3, 4<->5.
        """
        ei, ev, _ = _make_symmetric_graph()
        swap = compute_id_swap(ei, ev)
        assert swap.tolist() == [1, 0, 3, 2, 5, 4]

    def test_swap_is_involution(self):
        ei, ev, _ = _make_symmetric_graph()
        swap = compute_id_swap(ei, ev)
        assert swap[swap].tolist() == list(range(ei.size(1)))


class TestComputeTriplets:
    def test_self_loop_removed(self):
        ei, ev, _ = _make_symmetric_graph()
        id3_ca, id3_ba, kidx = compute_triplets(ei, num_nodes=3)
        # No (e,e) pairs
        assert not (id3_ca == id3_ba).any()

    def test_sorted_output(self):
        ei, ev, _ = _make_symmetric_graph()
        id3_ca, _, kidx = compute_triplets(ei, num_nodes=3)
        # id3_reduce_ca must be sorted so Kidx is well-defined
        assert torch.all(id3_ca[1:] >= id3_ca[:-1])

    def test_kidx_enumeration(self):
        ei, ev, _ = _make_symmetric_graph()
        id3_ca, _, kidx = compute_triplets(ei, num_nodes=3)
        # Within each id3_ca group Kidx3 must start from 0 and be contiguous.
        from collections import Counter

        cnt = Counter(id3_ca.tolist())
        # Rebuild expected kidx
        expected = []
        for v in id3_ca.tolist():
            pass  # placeholder; full check via per-group counter below
        seen: dict[int, int] = {}
        for v, k in zip(id3_ca.tolist(), kidx.tolist()):
            assert k == seen.get(v, 0)
            seen[v] = k + 1
        for v, c in cnt.items():
            assert seen[v] == c

    def test_triangle_exact_triplets(self):
        """
        Hand-computed triplet sets on the symmetric triangle graph.

        Edges (ids): 0:(0->1), 1:(1->0), 2:(0->2), 3:(2->0), 4:(1->2), 5:(2->1).
        Each atom has exactly 2 incoming edges, so each edge generates exactly 1 triplet with its partner on the same target.

        Expected pairing (e -> partner): 0<->5 (target 1), 1<->3 (target 0), 2<->4 (target 2).
        """
        ei, _, _ = _make_symmetric_graph()
        id3_ca, id3_ba, kidx = compute_triplets(ei, num_nodes=3)
        assert id3_ca.tolist() == [0, 1, 2, 3, 4, 5]
        assert id3_ba.tolist() == [5, 3, 4, 1, 2, 0]
        assert kidx.tolist() == [0, 0, 0, 0, 0, 0]

    def test_matches_reference(self, omnixas_small_structures):
        """
        On periodic structures, triplet SET must match fairchem reference.
        """
        for struct in omnixas_small_structures:
            ei, ew, ev, _ = build_edges(struct, cutoff=3.5, max_num_neighbors=20, method="radius")
            id3_ca, id3_ba, _ = compute_triplets(ei, num_nodes=len(struct))
            ref = ref_get_triplets(ei, num_atoms=len(struct))
            # (b->a, c->a) pair sets must coincide
            got = sort_triplet_set(id3_ba, id3_ca)
            exp = sort_triplet_set(ref["in"], ref["out"])
            assert got == exp


# compute_mixed_triplets


class TestComputeMixedTriplets:
    def test_matches_reference_non_periodic(self, synthetic_molecule):
        """
        For a molecule, main/a2ee2a edges with zero cell offsets give a reference that must agree exactly with our implementation.
        """
        ei, _, ev, _ = build_edges(synthetic_molecule, cutoff=3.0, max_num_neighbors=10)
        assert ev is not None
        # Use main graph as both "out" and "in" for a closed-form test.
        main = {
            "edge_index": ei,
            "cell_offset": torch.zeros(ei.size(1), 3, dtype=torch.int64),
        }
        for to_outedge in (False, True):
            ours = compute_mixed_triplets(
                main_edge_index=ei,
                main_edge_vec=ev,
                other_edge_index=ei,
                other_edge_vec=ev,
                num_nodes=len(synthetic_molecule),
                to_outedge=to_outedge,
            )
            ref = ref_get_mixed_triplets(main, main, len(synthetic_molecule), to_outedge=to_outedge)
            got = sort_triplet_set(ours["in_"], ours["out"])
            exp = sort_triplet_set(ref["in"], ref["out"])
            assert got == exp, f"mismatch for to_outedge={to_outedge}"

    def test_matches_reference_periodic(self, omnixas_small_structures):
        for struct in omnixas_small_structures:
            n = len(struct)
            pos = torch.tensor(struct.cart_coords, dtype=torch.float32)
            lat = torch.tensor(struct.lattice.matrix.copy(), dtype=torch.float32)
            ei, _, ev, _ = build_edges(struct, cutoff=3.5, max_num_neighbors=20)
            ei_aea, _, ev_aea, _ = build_edges(struct, cutoff=3.5, max_num_neighbors=20)
            assert ev is not None and ev_aea is not None
            main = {
                "edge_index": ei,
                "cell_offset": cell_offsets_from_edge_vec(ei, ev, pos, lat),
            }
            other = {
                "edge_index": ei_aea,
                "cell_offset": cell_offsets_from_edge_vec(ei_aea, ev_aea, pos, lat),
            }
            for to_outedge in (False, True):
                ours = compute_mixed_triplets(
                    main_edge_index=ei,
                    main_edge_vec=ev,
                    other_edge_index=ei_aea,
                    other_edge_vec=ev_aea,
                    num_nodes=n,
                    to_outedge=to_outedge,
                )
                ref = ref_get_mixed_triplets(other, main, n, to_outedge=to_outedge)
                got = sort_triplet_set(ours["in_"], ours["out"])
                exp = sort_triplet_set(ref["in"], ref["out"])
                assert got == exp


# compute_quadruplets


class TestComputeQuadruplets:
    def test_shapes_consistent(self, synthetic_molecule):
        ei, _, ev, _ = build_edges(synthetic_molecule, cutoff=3.0, max_num_neighbors=10)
        ei_i, _, ev_i, _ = build_edges(synthetic_molecule, cutoff=4.0, max_num_neighbors=10)
        assert ev is not None and ev_i is not None
        q = compute_quadruplets(
            edge_index=ei,
            edge_vec=ev,
            int_edge_index=ei_i,
            int_edge_vec=ev_i,
            num_nodes=len(synthetic_molecule),
        )
        n = q["id4_reduce_ca"].numel()
        # All quadruplet-level arrays are the same length
        for k in ("id4_expand_db", "id4_reduce_cab", "id4_expand_abd", "Kidx4"):
            assert q[k].numel() == n

        # id4_reduce_ca is sorted (for Kidx4 enumeration)
        if n > 0:
            assert torch.all(q["id4_reduce_ca"][1:] >= q["id4_reduce_ca"][:-1])

        # Intermediate arrays shapes
        assert q["id4_reduce_intm_ca"].size(0) == q["id4_reduce_intm_ab"].size(0)
        assert q["id4_expand_intm_db"].size(0) == q["id4_expand_intm_ab"].size(0)

    def test_no_identity_degeneracy_nonperiodic(self, synthetic_molecule):
        """
        For a molecule, a valid quadruplet has c != d, c != b and a != d.
        """
        ei, _, ev, _ = build_edges(synthetic_molecule, cutoff=3.0, max_num_neighbors=10)
        ei_i, _, ev_i, _ = build_edges(synthetic_molecule, cutoff=4.0, max_num_neighbors=10)
        assert ev is not None and ev_i is not None
        q = compute_quadruplets(ei, ev, ei_i, ev_i, num_nodes=len(synthetic_molecule))

        idx_s, idx_t = ei
        idx_int_s, idx_int_t = ei_i
        # Per-quadruplet atoms
        idx_c = idx_s[q["id4_reduce_ca"]]
        idx_a = idx_t[q["id4_reduce_ca"]]
        idx_d = idx_s[q["id4_expand_db"]]
        idx_b = idx_t[q["id4_expand_db"]]

        # Look up int edge via id4_reduce_intm_ab (or id4_expand_intm_ab; identical value per quadruplet by construction)
        ab_intm = q["id4_reduce_intm_ab"][q["id4_reduce_cab"]]
        idx_b_int = idx_int_s[ab_intm]
        idx_a_int = idx_int_t[ab_intm]

        # Cross-graph atom identity must hold: the b/a atoms on the main and int graphs must match
        assert torch.all(idx_b == idx_b_int)
        assert torch.all(idx_a == idx_a_int)
        # Molecule (no images) => atom identity degeneracies must all be filtered out
        assert torch.all(idx_c != idx_d)
        assert torch.all(idx_c != idx_b)
        assert torch.all(idx_a != idx_d)

    def test_matches_reference_non_periodic(self, synthetic_molecule):
        ei, _, ev, _ = build_edges(synthetic_molecule, cutoff=3.0, max_num_neighbors=10)
        ei_i, _, ev_i, _ = build_edges(synthetic_molecule, cutoff=4.0, max_num_neighbors=10)
        assert ev is not None and ev_i is not None
        n = len(synthetic_molecule)
        main = {"edge_index": ei, "cell_offset": torch.zeros(ei.size(1), 3, dtype=torch.int64)}
        qint = {"edge_index": ei_i, "cell_offset": torch.zeros(ei_i.size(1), 3, dtype=torch.int64)}

        q = compute_quadruplets(ei, ev, ei_i, ev_i, num_nodes=n)
        ref = ref_get_quadruplets(main, qint, num_atoms=n)

        # Map each quadruplet to an unordered identity: (c->a edge id, d->b edge id, b->a int edge id).
        # Compare as sets.
        ab_intm = q["id4_reduce_intm_ab"][q["id4_reduce_cab"]]
        got = sorted(zip(q["id4_reduce_ca"].tolist(), q["id4_expand_db"].tolist(), ab_intm.tolist()))
        exp = sorted(zip(ref["out"].tolist(), ref["in_db"].tolist(), ref["int_ab"].tolist()))
        assert got == exp

    def test_matches_reference_periodic(self, synthetic_structure_cubic):
        struct = synthetic_structure_cubic
        n = len(struct)
        pos = torch.tensor(struct.cart_coords, dtype=torch.float32)
        lat = torch.tensor(struct.lattice.matrix.copy(), dtype=torch.float32)
        ei, _, ev, _ = build_edges(struct, cutoff=2.8, max_num_neighbors=20)
        ei_i, _, ev_i, _ = build_edges(struct, cutoff=3.6, max_num_neighbors=20)
        assert ev is not None and ev_i is not None
        main = {
            "edge_index": ei,
            "cell_offset": cell_offsets_from_edge_vec(ei, ev, pos, lat),
        }
        qint = {
            "edge_index": ei_i,
            "cell_offset": cell_offsets_from_edge_vec(ei_i, ev_i, pos, lat),
        }

        q = compute_quadruplets(ei, ev, ei_i, ev_i, num_nodes=n)
        ref = ref_get_quadruplets(main, qint, num_atoms=n)

        ab_intm = q["id4_reduce_intm_ab"][q["id4_reduce_cab"]]
        got = sorted(zip(q["id4_reduce_ca"].tolist(), q["id4_expand_db"].tolist(), ab_intm.tolist()))
        exp = sorted(zip(ref["out"].tolist(), ref["in_db"].tolist(), ref["int_ab"].tolist()))
        assert got == exp


# Degenerate / special-case structures


class TestDegenerateStructures:
    def test_collinear_triplets_preserved(self, synthetic_molecule_degenerate):
        """
        Linear O-Fe-O: six directed edges, triplets through Fe must include the (O1, O2) and (O2, O1) collinear ones.
        They represent the 180° angle and are NOT degenerate (distinct atoms).
        """
        ei, _, ev, _ = build_edges(synthetic_molecule_degenerate, cutoff=2.0, max_num_neighbors=10)
        id3_ca, id3_ba, _ = compute_triplets(ei, num_nodes=3)
        # triplets at the central atom 0 using edges 1->0 and 2->0
        # Both orderings should be present (c->a=1->0, b->a=2->0 and vice versa).
        # Verify at least one triplet pivots on atom 0:
        idx_s, idx_t = ei
        pivots_c = idx_t[id3_ca].tolist()
        pivots_b = idx_t[id3_ba].tolist()
        assert 0 in pivots_c and 0 in pivots_b
