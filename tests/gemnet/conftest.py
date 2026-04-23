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
Pytest fixtures and fairchem-style reference implementations used by the GemNet / GemNet-OC test suite.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Molecule, Structure
from torch_sparse import SparseTensor

# Make the repository root importable without installing.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Helpers


def cell_offsets_from_edge_vec(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    positions: torch.Tensor,
    lattice: torch.Tensor | None,
) -> torch.Tensor:
    """
    Recover integer lattice-offset vectors (``k``) such that

        ``edge_vec[e] = (pos[t] + k @ lattice) - pos[s]``.

    For non-periodic inputs (``lattice is None``) every offset is zero.
    """
    if edge_index.size(1) == 0:
        return torch.zeros(0, 3, dtype=torch.int64)
    src = edge_index[0]
    dst = edge_index[1]
    dvec = edge_vec - (positions[dst] - positions[src])
    if lattice is None:
        return torch.zeros(edge_index.size(1), 3, dtype=torch.int64)
    L_inv = torch.linalg.inv(lattice.to(dvec.dtype))
    offsets = (dvec @ L_inv).round().to(torch.int64)
    return offsets


def sort_triplet_set(
    t_in: torch.Tensor,
    t_out: torch.Tensor,
) -> list[tuple[int, int]]:
    """
    Sort (in, out) triplet pairs lexicographically for set-equality checks.
    """
    return sorted(zip(t_in.tolist(), t_out.tolist()))


# Independent periodic-neighbor reference.


def reference_periodic_edges(
    structure: Structure,
    cutoff: float,
    image_range: int = 2,
) -> set[tuple[int, int, tuple[int, int, int]]]:
    """
    Enumerate all directed edges (src, dst, cell_offset) within ``cutoff`` by iterating over every atom pair
    and every integer lattice offset ``k = (a, b, c)`` with ``|a|, |b|, |c| <= image_range``.

    Self-edges with zero displacement are excluded. The returned set is keyed by the *integer* cell offset,
    which is the canonical fairchem representation — it is a cleaner comparison target than rounded floats.
    """
    L = np.asarray(structure.lattice.matrix, dtype=np.float64)
    pos = np.asarray(structure.cart_coords, dtype=np.float64)
    n = len(structure)
    edges: set[tuple[int, int, tuple[int, int, int]]] = set()
    for i in range(n):
        for j in range(n):
            for a in range(-image_range, image_range + 1):
                for b in range(-image_range, image_range + 1):
                    for c in range(-image_range, image_range + 1):
                        off = np.array([a, b, c], dtype=np.float64) @ L
                        vec = pos[j] + off - pos[i]
                        d = float(np.linalg.norm(vec))
                        if 0.0 < d <= cutoff + 1e-6:
                            edges.add((i, j, (a, b, c)))
    return edges


def reference_periodic_edges_with_vecs(
    structure: Structure,
    cutoff: float,
    image_range: int = 2,
) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
    """
    Same as ``reference_periodic_edges`` but also returns the explicit displacement vectors for each enumerated edge.
    """
    L = np.asarray(structure.lattice.matrix, dtype=np.float64)
    pos = np.asarray(structure.cart_coords, dtype=np.float64)
    n = len(structure)
    edges: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
    for i in range(n):
        for j in range(n):
            for a in range(-image_range, image_range + 1):
                for b in range(-image_range, image_range + 1):
                    for c in range(-image_range, image_range + 1):
                        off = np.array([a, b, c], dtype=np.float64) @ L
                        vec = pos[j] + off - pos[i]
                        d = float(np.linalg.norm(vec))
                        if 0.0 < d <= cutoff + 1e-6:
                            edges[(i, j, (a, b, c))] = vec
    return edges


# Fairchem-style reference implementations


def ref_get_triplets(edge_index: torch.Tensor, num_atoms: int) -> dict[str, torch.Tensor]:
    """
    Port of fairchem ``get_triplets``.

    Filters only by edge-id equality (``in == out``) which is correct forboth periodic and
    non-periodic inputs (duplicate images have distinct edge ids).
    """
    idx_s, idx_t = edge_index
    n_edges = idx_s.size(0)
    value = torch.arange(n_edges, device=idx_s.device, dtype=idx_s.dtype)
    adj = SparseTensor(row=idx_t, col=idx_s, value=value, sparse_sizes=(num_atoms, num_atoms))
    adj_sel = adj[idx_t]  # type: ignore[index]
    in_ = adj_sel.storage.value()
    out = adj_sel.storage.row()
    mask = in_ != out
    return {"in": in_[mask], "out": out[mask]}


def ref_get_mixed_triplets(
    graph_in: dict[str, torch.Tensor],
    graph_out: dict[str, torch.Tensor],
    num_atoms: int,
    to_outedge: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Port of ``get_mixed_triplets`` (cell-offset degeneracy test).
    """
    idx_out_s, idx_out_t = graph_out["edge_index"]
    idx_in_s, idx_in_t = graph_in["edge_index"]

    value_in = torch.arange(idx_in_s.size(0), device=idx_in_s.device, dtype=idx_in_s.dtype)
    adj_in = SparseTensor(row=idx_in_t, col=idx_in_s, value=value_in, sparse_sizes=(num_atoms, num_atoms))
    adj_edges = adj_in[idx_out_s] if to_outedge else adj_in[idx_out_t]  # type: ignore[index]

    idx_in = adj_edges.storage.value()
    idx_out = adj_edges.storage.row()

    if to_outedge:
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_t[idx_out]
        cell_sum = graph_out["cell_offset"][idx_out] + graph_in["cell_offset"][idx_in]
    else:
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_s[idx_out]
        cell_sum = graph_out["cell_offset"][idx_out] - graph_in["cell_offset"][idx_in]

    mask = (idx_atom_in != idx_atom_out) | torch.any(cell_sum != 0, dim=-1)
    return {"in": idx_in[mask], "out": idx_out[mask]}


def ref_get_quadruplets(
    main_graph: dict[str, torch.Tensor],
    qint_graph: dict[str, torch.Tensor],
    num_atoms: int,
) -> dict[str, torch.Tensor]:
    """
    Port of ``get_quadruplets``. Returns ``out`` and the two triplet sets used for angle computation.

    We only need the identity of quadruplets (``c->a<-b->d`` mapped back to main/qint edge ids)
    for correctness comparisons, so we keep the minimum set of outputs.
    """
    trip_in = ref_get_mixed_triplets(main_graph, qint_graph, num_atoms, to_outedge=True)
    trip_out = ref_get_mixed_triplets(qint_graph, main_graph, num_atoms, to_outedge=False)

    # For each triplet_out record (triplet c->a<-b), expand by the matching
    # d->b->a edges from triplet_in.
    # triplet_out["in"] = interaction-edge id (b->a);
    # triplet_in ["out"] = interaction-edge id (b->a).
    # For each b->a int edge, count number of d->b triplets attached.
    n_int = qint_graph["edge_index"].size(1)
    n_trip_in_per_int = torch.bincount(trip_in["out"], minlength=n_int)
    n_trip_out_per_trip_out_elem = n_trip_in_per_int[trip_out["in"]]

    # Repeat each triplet_out by the count of matching triplet_ins
    out_ca = torch.repeat_interleave(trip_out["out"], n_trip_out_per_trip_out_elem)
    idx_inter = torch.repeat_interleave(trip_out["in"], n_trip_out_per_trip_out_elem)

    # The matching trip_in "in" edges, in int-edge order. We need an ordering
    # consistent with fairchem. Construct the adjacency from trip_in indexed by int edge.
    # Build list: for each int edge i, the list of d->b edge ids attached.
    order = torch.argsort(trip_in["out"], stable=True)
    in_sorted = trip_in["in"][order]
    out_sorted = trip_in["out"][order]
    # Cumulative offsets per int edge
    offsets = torch.cumsum(n_trip_in_per_int, 0) - n_trip_in_per_int
    # For each expanded quadruplet: find its position within the sorted
    # per-int-edge list. Within trip_out expanded, consecutive entries share
    # the same idx_inter; enumerate 0..count-1.
    inner = torch.arange(out_ca.size(0), device=out_ca.device)
    # Compute group start per expanded row
    # group_ids = idx_inter; inner index within group via shifting by cumcount.
    # Use manual: sort by idx_inter (stable) and enumerate.
    arg_group = torch.argsort(idx_inter, stable=True)
    counts_per_inter = torch.bincount(idx_inter, minlength=n_int)
    group_starts = torch.cumsum(counts_per_inter, 0) - counts_per_inter
    pos_sorted = torch.arange(out_ca.size(0), device=out_ca.device)
    inner_sorted = pos_sorted - group_starts[idx_inter[arg_group]]
    inner = torch.empty_like(idx_inter)
    inner[arg_group] = inner_sorted

    # For each expansion the inner index cycles through the trip_in group;
    # consecutive expansions sharing the same idx_inter come in groups of
    # ``n_trip_in_per_int[idx_inter]`` trip_out elements.
    inner_mod = inner % n_trip_in_per_int[idx_inter]
    in_db_per_quad = in_sorted[offsets[idx_inter] + inner_mod]

    # Degeneracy test: c == d at same image.
    # c = main_src[trip_out["out"]][...], d = main_src[in_db_per_quad]
    idx_s_main = main_graph["edge_index"][0]
    idx_atom_c = idx_s_main[out_ca]
    idx_atom_d = idx_s_main[in_db_per_quad]
    cell_offset_cd = (
        main_graph["cell_offset"][in_db_per_quad]
        + qint_graph["cell_offset"][idx_inter]
        - main_graph["cell_offset"][out_ca]
    )
    mask_cd = (idx_atom_c != idx_atom_d) | torch.any(cell_offset_cd != 0, dim=-1)

    return {
        "out": out_ca[mask_cd],  # main edge id (c->a) per quadruplet
        "in_db": in_db_per_quad[mask_cd],  # main edge id (d->b) per quadruplet
        "int_ab": idx_inter[mask_cd],  # interaction edge id (b->a) per quadruplet
    }


# Molecule / Structure fixtures


def _attach_xanes(obj, absorber_indices: list[int], n_energies: int = 16):
    """
    Stamp synthetic XANES dictionaries on selected site indices.
    """
    spectra: list[dict | None] = [None] * len(obj)
    rng = np.random.default_rng(0)
    for i in absorber_indices:
        spectra[i] = {
            "energies": rng.uniform(7100.0, 7200.0, size=n_energies).astype(np.float32),
            "intensities": rng.uniform(0.0, 1.0, size=n_energies).astype(np.float32),
        }
    obj.add_site_property("XANES", spectra)
    obj.properties["file_name"] = "test_obj"
    return obj


@pytest.fixture(scope="session")
def synthetic_molecule():
    """
    Non-periodic Fe(CO)4 cluster with XANES on the absorber only.
    """
    species = ["Fe", "O", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.0],
        [1.80, 0.0, 0.0],
        [-1.80, 0.0, 0.0],
        [0.0, 1.80, 0.0],
        [0.0, 0.0, 1.80],
    ]
    mol = Molecule(species, coords)
    return _attach_xanes(mol, [0])


@pytest.fixture(scope="session")
def synthetic_molecule_degenerate():
    """
    Linear triatomic with three collinear atoms — probes 0° / 180° triplets.
    """
    species = ["Fe", "O", "O"]
    coords = [
        [0.0, 0.0, 0.0],
        [1.50, 0.0, 0.0],
        [-1.50, 0.0, 0.0],
    ]
    mol = Molecule(species, coords)
    return _attach_xanes(mol, [0])


@pytest.fixture(scope="session")
def synthetic_structure_cubic():
    """
    Small cubic lattice whose cutoff exceeds the cell edge — stress PBC.
    """
    species = ["Fe", "O"]
    lattice = Lattice.cubic(3.0)
    coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    s = Structure(lattice, species, coords)
    return _attach_xanes(s, [0])


# Bundled example inputs live under tests/gemnet/data/ so the suite is
# self-contained (no dependency on absolute paths outside the repo).
_DATA_DIR = Path(__file__).resolve().parent / "data"
_OMNIXAS_DIR = _DATA_DIR / "omnixas"
_FE_XYZ_DIR = _DATA_DIR / "fe" / "xyz"
_FE_XANES_DIR = _DATA_DIR / "fe" / "xanes"


@pytest.fixture(scope="session")
def omnixas_data_dir() -> Path:
    """
    Bundled periodic example inputs (pymatgen Structure JSONs).
    """
    assert _OMNIXAS_DIR.is_dir(), f"Bundled omnixas data missing: {_OMNIXAS_DIR}"
    return _OMNIXAS_DIR


@pytest.fixture(scope="session")
def omnixas_small_structures(omnixas_data_dir) -> list[Structure]:
    """
    All bundled periodic omnixas structures (>= 10).
    """
    structs: list[Structure] = []
    for p in sorted(omnixas_data_dir.glob("*.json")):
        s = Structure.from_file(str(p))
        s.properties.setdefault("file_name", p.stem)
        structs.append(s)
    assert len(structs) >= 10, f"Need >=10 periodic test structures, found {len(structs)}"
    return structs


@pytest.fixture(scope="session")
def omnixas_temp_data_dir(omnixas_data_dir, tmp_path_factory) -> Path:
    """
    Copy the bundled omnixas JSONs into a temp dir so a dataset can scan a writable location.
    """
    dst = tmp_path_factory.mktemp("omnixas_mini")
    for src in omnixas_data_dir.glob("*.json"):
        shutil.copy(src, dst / src.name)
    return dst


@pytest.fixture(scope="session")
def fe_mini_data(tmp_path_factory) -> tuple[Path, Path]:
    """
    Bundled non-periodic Fe complexes (xyz + XANES, >= 10 structures).
    """
    assert _FE_XYZ_DIR.is_dir() and _FE_XANES_DIR.is_dir(), f"Bundled fe data missing under {_DATA_DIR}"
    dst_root = tmp_path_factory.mktemp("fe_mini")
    dst_xyz = dst_root / "xyz_train"
    dst_xanes = dst_root / "xanes_train"
    dst_xyz.mkdir()
    dst_xanes.mkdir()
    picked = 0
    for xyz in sorted(_FE_XYZ_DIR.glob("*.xyz")):
        xan = _FE_XANES_DIR / (xyz.stem + ".txt")
        if xan.exists():
            shutil.copy(xyz, dst_xyz / xyz.name)
            shutil.copy(xan, dst_xanes / xan.name)
            picked += 1
    assert picked >= 10, f"Need >=10 fe test molecules, found {picked}"
    return dst_xyz, dst_xanes
