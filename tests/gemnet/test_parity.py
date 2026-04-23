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
Parity tests comparing xanesnet implementations against independent reference implementations 
(fairchem ports or brute-force oracles).
"""

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from xanesnet.utils.graph import build_edges
from xanesnet.utils.graph.gemnet_indices import (
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
    reference_periodic_edges,
    reference_periodic_edges_with_vecs,
    sort_triplet_set,
)

# Fixtures local to the parity suite (kept small so tests run fast).


def _cubic_fe_o() -> Structure:
    """
    Cubic a=3.0 Fe/O structure. Cutoff > cell edge -> rich PBC behaviour.
    """
    return Structure(Lattice.cubic(3.0), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _orthorhombic() -> Structure:
    """
    Non-cubic cell to catch lattice-inversion bugs.
    """
    lat = Lattice.from_parameters(3.5, 4.2, 5.1, 90, 90, 90)
    return Structure(lat, ["Fe", "O", "O"], [[0, 0, 0], [0.3, 0.1, 0.2], [0.7, 0.8, 0.6]])


def _triclinic() -> Structure:
    """
    Non-orthogonal cell — stresses fractional <-> cartesian conversion.
    """
    lat = Lattice.from_parameters(4.0, 4.2, 3.8, 85, 95, 100)
    return Structure(lat, ["Fe", "O"], [[0, 0, 0], [0.4, 0.3, 0.5]])


def _bcc_fe() -> Structure:
    """
    Body-centered cubic iron (two atoms).
    """
    return Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _fcc_cu() -> Structure:
    """
    Face-centered cubic copper (four-atom conventional cell).
    """
    return Structure(
        Lattice.cubic(3.615),
        ["Cu"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


def _hexagonal_feo() -> Structure:
    """
    Hexagonal cell with non-90 gamma angle.
    """
    lat = Lattice.hexagonal(3.2, 5.1)
    return Structure(lat, ["Fe", "O"], [[0, 0, 0], [1 / 3, 2 / 3, 0.5]])


def _rutile() -> Structure:
    """
    Rutile-like tetragonal cell (Fe + O octahedra).
    """
    lat = Lattice.tetragonal(4.6, 2.95)
    return Structure(
        lat,
        ["Fe", "Fe", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.3, 0.3, 0],
            [0.7, 0.7, 0],
            [0.8, 0.2, 0.5],
            [0.2, 0.8, 0.5],
        ],
    )


def _monoclinic() -> Structure:
    """
    Monoclinic cell (beta != 90).
    """
    lat = Lattice.from_parameters(5.0, 4.5, 6.1, 90, 102, 90)
    return Structure(lat, ["Fe", "O", "O", "C"], [[0, 0, 0], [0.25, 0.5, 0.25], [0.75, 0.5, 0.75], [0.4, 0.1, 0.6]])


def _large_cubic() -> Structure:
    """
    Larger cubic cell with 8 atoms and mixed species.
    """
    return Structure(
        Lattice.cubic(5.5),
        ["Fe", "Fe", "O", "O", "O", "O", "N", "C"],
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.1, 0.4, 0.1],
            [0.6, 0.1, 0.4],
        ],
    )


def _skew_triclinic() -> Structure:
    """
    Heavily skewed triclinic cell — all three angles off-90.
    """
    lat = Lattice.from_parameters(3.7, 3.9, 4.5, 78, 88, 112)
    return Structure(lat, ["Fe", "O", "N"], [[0, 0, 0], [0.4, 0.2, 0.5], [0.7, 0.8, 0.3]])


def _small_mixed() -> Structure:
    """
    Small mixed transition metal + chalcogen cell.
    """
    lat = Lattice.from_parameters(3.3, 3.3, 5.2, 90, 90, 120)
    return Structure(lat, ["Fe", "S"], [[0, 0, 0], [1 / 3, 2 / 3, 0.5]])


PERIODIC_FIXTURES = [
    _cubic_fe_o,
    _orthorhombic,
    _triclinic,
    _bcc_fe,
    _fcc_cu,
    _hexagonal_feo,
    _rutile,
    _monoclinic,
    _large_cubic,
    _skew_triclinic,
    _small_mixed,
]
assert len(PERIODIC_FIXTURES) >= 10


# Non-periodic (molecule) fixtures for parity tests that do not require a lattice. We need >= 10 different geometries.
from pymatgen.core import Molecule  # noqa: E402


def _mol_fe_octahedral() -> Molecule:
    return Molecule(
        ["Fe"] + ["O"] * 6, [[0, 0, 0], [2.0, 0, 0], [-2.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0], [0, 0, 2.0], [0, 0, -2.0]]
    )


def _mol_fe_tetrahedral() -> Molecule:
    return Molecule(
        ["Fe", "Cl", "Cl", "Cl", "Cl"],
        [[0, 0, 0], [1.3, 1.3, 1.3], [-1.3, -1.3, 1.3], [-1.3, 1.3, -1.3], [1.3, -1.3, -1.3]],
    )


def _mol_fe_square_planar() -> Molecule:
    return Molecule(["Fe", "N", "N", "N", "N"], [[0, 0, 0], [2.0, 0, 0], [-2.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0]])


def _mol_ferrocene_like() -> Molecule:
    coords: list[list[float]] = [[0.0, 0.0, 0.0]]
    for i in range(5):
        a = i * 2 * np.pi / 5
        coords.append([1.2 * np.cos(a), 1.2 * np.sin(a), 1.6])
    for i in range(5):
        a = i * 2 * np.pi / 5 + np.pi / 5
        coords.append([1.2 * np.cos(a), 1.2 * np.sin(a), -1.6])
    return Molecule(["Fe"] + ["C"] * 10, coords)


def _mol_linear() -> Molecule:
    return Molecule(["Fe", "O", "O"], [[0, 0, 0], [1.8, 0, 0], [-1.8, 0, 0]])


def _mol_trigonal_bipyr() -> Molecule:
    return Molecule(
        ["Fe", "N", "N", "N", "O", "O"],
        [[0, 0, 0], [2.0, 0, 0], [-1.0, 1.73, 0], [-1.0, -1.73, 0], [0, 0, 2.0], [0, 0, -2.0]],
    )


def _mol_water_cluster() -> Molecule:
    return Molecule(
        ["Fe", "O", "H", "H", "O", "H", "H"],
        [[0, 0, 0], [2.2, 0, 0], [2.7, 0.8, 0], [2.7, -0.8, 0], [-2.2, 0, 0], [-2.7, 0.8, 0], [-2.7, -0.8, 0]],
    )


def _mol_bent() -> Molecule:
    return Molecule(["Fe", "C", "N"], [[0, 0, 0], [1.8, 0.3, 0], [3.0, 0.6, 0]])


def _mol_cyclopentadienyl() -> Molecule:
    coords: list[list[float]] = [[0.0, 0.0, 0.0]]
    for i in range(5):
        a = i * 2 * np.pi / 5
        coords.append([1.2 * np.cos(a), 1.2 * np.sin(a), 1.8])
    return Molecule(["Fe"] + ["C"] * 5, coords)


def _mol_small_chain() -> Molecule:
    return Molecule(["Fe", "C", "C", "O"], [[0, 0, 0], [1.8, 0, 0], [3.1, 0, 0], [4.3, 0, 0]])


MOLECULE_FIXTURES = [
    _mol_fe_octahedral,
    _mol_fe_tetrahedral,
    _mol_fe_square_planar,
    _mol_ferrocene_like,
    _mol_linear,
    _mol_trigonal_bipyr,
    _mol_water_cluster,
    _mol_bent,
    _mol_cyclopentadienyl,
    _mol_small_chain,
]
assert len(MOLECULE_FIXTURES) >= 10


pytestmark = pytest.mark.parity


# 1. PERIODIC EDGE-BUILDING PARITY  (top priority)


@pytest.mark.parametrize("mk_struct", PERIODIC_FIXTURES)
@pytest.mark.parametrize("cutoff", [2.5, 3.5, 4.5])
def test_periodic_edges_match_brute_force(mk_struct, cutoff):
    """
    ``build_edges`` on a periodic Structure must produce exactly the same directed edges
    (keyed by source atom, target atom, integer lattice offset) as a brute-force image enumeration.

    This is the single most important correctness check for PBC data preparation.
    """
    s = mk_struct()
    ei, ew, ev, _ = build_edges(s, cutoff=cutoff, max_num_neighbors=1000, method="radius")
    assert ev is not None
    # Recover integer lattice offsets from the returned edge vectors.
    pos = torch.tensor(s.cart_coords, dtype=torch.float64)
    lattice = torch.tensor(s.lattice.matrix, dtype=torch.float64)
    offsets = cell_offsets_from_edge_vec(ei, ev.to(torch.float64), pos, lattice)
    xn_edges = {(int(ei[0, k]), int(ei[1, k]), tuple(offsets[k].tolist())) for k in range(ei.size(1))}
    ref_edges = reference_periodic_edges(s, cutoff=cutoff, image_range=2)
    assert xn_edges == ref_edges, (
        f"Edge mismatch (cutoff={cutoff}). " f"xanesnet-only: {xn_edges - ref_edges}, ref-only: {ref_edges - xn_edges}"
    )
    # edge_weight must equal ||edge_vec|| exactly.
    assert torch.allclose(ew, torch.linalg.norm(ev, dim=-1), atol=1e-5)


@pytest.mark.parametrize("mk_struct", PERIODIC_FIXTURES)
def test_periodic_edge_vecs_match_analytic(mk_struct):
    """
    The displacement vector of every edge must equal ``p_dst + offset @ L - p_src`` to floating-point precision.
    """
    s = mk_struct()
    cutoff = 3.5
    ei, _, ev, _ = build_edges(s, cutoff=cutoff, max_num_neighbors=1000, method="radius")
    assert ev is not None
    ref = reference_periodic_edges_with_vecs(s, cutoff=cutoff, image_range=2)
    pos = torch.tensor(s.cart_coords, dtype=torch.float64)
    lattice = torch.tensor(s.lattice.matrix, dtype=torch.float64)
    offsets = cell_offsets_from_edge_vec(ei, ev.to(torch.float64), pos, lattice)
    for k in range(ei.size(1)):
        key = (int(ei[0, k]), int(ei[1, k]), tuple(offsets[k].tolist()))
        expected = ref[key]
        got = ev[k].to(torch.float64).numpy()
        assert np.allclose(got, expected, atol=1e-5), f"edge_vec mismatch for {key}: got {got} vs expected {expected}"


@pytest.mark.parametrize("mk_struct", PERIODIC_FIXTURES)
def test_periodic_edges_bidirectional(mk_struct):
    """
    For every directed edge (src, dst, k), the reverse (dst, src, -k) must also be present;
    a fairchem invariant that our pipeline relies on.
    """
    s = mk_struct()
    ei, _, ev, _ = build_edges(s, cutoff=3.5, max_num_neighbors=1000, method="radius")
    assert ev is not None
    pos = torch.tensor(s.cart_coords, dtype=torch.float64)
    lattice = torch.tensor(s.lattice.matrix, dtype=torch.float64)
    offsets = cell_offsets_from_edge_vec(ei, ev.to(torch.float64), pos, lattice)
    edge_set = {(int(ei[0, k]), int(ei[1, k]), tuple(offsets[k].tolist())) for k in range(ei.size(1))}
    for src, dst, k in edge_set:
        reverse = (dst, src, (-k[0], -k[1], -k[2]))
        assert reverse in edge_set, f"Missing reverse edge for {(src, dst, k)}"

    # id_swap must be a valid involution pairing each edge to its reverse.
    swap = compute_id_swap(ei, ev)
    assert swap[swap].tolist() == list(range(ei.size(1)))
    # edge_vec must negate under swap.
    assert torch.allclose(ev[swap], -ev, atol=1e-5)


# 2. TRIPLET / QUADRUPLET PARITY ON PERIODIC INPUTS
#
# The xanesnet functions (vec-based PBC degeneracy) must produce the same set of triplets / quadruplets
# as the fairchem reference (cell_offset-based) on the same periodic edge set.


@pytest.mark.parametrize("mk_struct", PERIODIC_FIXTURES)
def test_triplets_match_fairchem(mk_struct):
    s = mk_struct()
    ei, _, ev, _ = build_edges(s, cutoff=3.5, max_num_neighbors=30, method="radius")
    n = len(s)
    id3_ca, id3_ba, _ = compute_triplets(ei, num_nodes=n)
    ref = ref_get_triplets(ei, num_atoms=n)
    assert sort_triplet_set(id3_ba, id3_ca) == sort_triplet_set(ref["in"], ref["out"])


@pytest.mark.molecule
@pytest.mark.parametrize("mk_mol", MOLECULE_FIXTURES)
def test_triplets_match_fairchem_molecules(mk_mol):
    """
    Same triplet-set parity, but on 10 non-periodic molecular geometries.
    """
    mol = mk_mol()
    ei, _, ev, _ = build_edges(mol, cutoff=3.5, max_num_neighbors=30, method="radius")
    n = len(mol)
    if ei.size(1) == 0:
        pytest.skip("no edges")
    id3_ca, id3_ba, _ = compute_triplets(ei, num_nodes=n)
    ref = ref_get_triplets(ei, num_atoms=n)
    assert sort_triplet_set(id3_ba, id3_ca) == sort_triplet_set(ref["in"], ref["out"])


@pytest.mark.parametrize("mk_struct", PERIODIC_FIXTURES)
def test_mixed_triplets_match_fairchem(mk_struct):
    s = mk_struct()
    ei_main, _, ev_main, _ = build_edges(s, cutoff=3.0, max_num_neighbors=30, method="radius")
    ei_other, _, ev_other, _ = build_edges(s, cutoff=4.0, max_num_neighbors=30, method="radius")
    assert ev_main is not None and ev_other is not None
    n = len(s)
    pos = torch.tensor(s.cart_coords, dtype=torch.float64)
    lattice = torch.tensor(s.lattice.matrix, dtype=torch.float64)
    cell_main = cell_offsets_from_edge_vec(ei_main, ev_main.to(torch.float64), pos, lattice)
    cell_other = cell_offsets_from_edge_vec(ei_other, ev_other.to(torch.float64), pos, lattice)

    for to_outedge in (True, False):
        out = compute_mixed_triplets(ei_main, ev_main, ei_other, ev_other, num_nodes=n, to_outedge=to_outedge)
        # xanesnet naming: ``main_edge_index`` holds the OUTPUT edges (theones iterated over),
        # ``other_edge_index`` holds the INPUT edges. The fairchem port uses the opposite naming (``graph_in`` = inputs).
        ref = ref_get_mixed_triplets(
            graph_in={"edge_index": ei_other, "cell_offset": cell_other},
            graph_out={"edge_index": ei_main, "cell_offset": cell_main},
            num_atoms=n,
            to_outedge=to_outedge,
        )
        # xanesnet uses the key ``in_`` to avoid shadowing the Python builtin; the fairchem port uses ``in``.
        assert sort_triplet_set(out["in_"], out["out"]) == sort_triplet_set(
            ref["in"], ref["out"]
        ), f"mixed triplets mismatch (to_outedge={to_outedge})"


@pytest.mark.parametrize("mk_struct", PERIODIC_FIXTURES)
def test_quadruplets_match_fairchem(mk_struct):
    s = mk_struct()
    ei_main, _, ev_main, _ = build_edges(s, cutoff=3.0, max_num_neighbors=30, method="radius")
    ei_int, _, ev_int, _ = build_edges(s, cutoff=4.0, max_num_neighbors=30, method="radius")
    assert ev_main is not None and ev_int is not None
    n = len(s)
    pos = torch.tensor(s.cart_coords, dtype=torch.float64)
    lattice = torch.tensor(s.lattice.matrix, dtype=torch.float64)
    cell_main = cell_offsets_from_edge_vec(ei_main, ev_main.to(torch.float64), pos, lattice)
    cell_int = cell_offsets_from_edge_vec(ei_int, ev_int.to(torch.float64), pos, lattice)

    out = compute_quadruplets(ei_main, ev_main, ei_int, ev_int, num_nodes=n)
    ref = ref_get_quadruplets(
        main_graph={"edge_index": ei_main, "cell_offset": cell_main},
        qint_graph={"edge_index": ei_int, "cell_offset": cell_int},
        num_atoms=n,
    )
    # ref_get_quadruplets returns (main_edge_id, int_edge_id) pairs for outgoing quadruplets.
    # xanesnet returns the same pairs as (id4_reduce_ca, id4_expand_db)
    # where id4_reduce_ca is the main-graph c->a edge and id4_expand_db the qint d->b edge.
    # xanesnet returns (id4_reduce_ca = main c->a, id4_expand_db = main d->b); fairchem port returns (out = main c->a, in_db = main d->b).
    xn_pairs = sorted(zip(out["id4_reduce_ca"].tolist(), out["id4_expand_db"].tolist()))
    ref_pairs = sorted(zip(ref["out"].tolist(), ref["in_db"].tolist()))
    assert xn_pairs == ref_pairs


# 3. RADIAL BASIS PARITY (GemNet-OC)


def test_radial_basis_matches_fairchem_port():
    """
    xanesnet's ``RadialBasis`` forward output must match the fairchem reference exactly for identical configuration and distances
    (there are no learnable parameters in the default config).
    """
    from xanesnet.models.gemnet_oc.layers.radial_basis import RadialBasis as XN_RB

    # Construct both: spherical_bessel + polynomial envelope -> no params.
    xn = XN_RB(
        num_radial=6,
        cutoff=6.0,
        rbf={"name": "spherical_bessel"},
        envelope={"name": "polynomial", "exponent": 5},
    )
    xn.eval()
    d = torch.linspace(0.05, 5.99, steps=20)
    out_xn = xn(d)
    # The fairchem RadialBasis depends on fairchem.core at import time, which may not be installed.
    # We therefore reconstruct its math locally from the formulas in the source:
    # spherical Bessel with canonical frequencies k_n = n * pi, scaled by sqrt(2 / cutoff) / d_scaled, and a
    # polynomial envelope with exponent 5. This is the *analytic* reference.
    import math

    num_radial = 6
    cutoff = 6.0
    d_scaled = d / cutoff
    # Polynomial envelope (exponent p=5)
    p = 5
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    env = 1 + a * d_scaled**p + b * d_scaled ** (p + 1) + c * d_scaled ** (p + 2)
    env = torch.where(d_scaled < 1, env, torch.zeros_like(env))
    # Spherical Bessel basis: frequencies initialised at n*pi for n=1..num_radial.
    freqs = torch.tensor([(n + 1) * np.pi for n in range(num_radial)])
    norm_const = math.sqrt(2.0 / (cutoff**3))
    bessel = norm_const / d_scaled[:, None] * torch.sin(freqs * d_scaled[:, None])
    out_ref = env[:, None] * bessel
    assert torch.allclose(
        out_xn, out_ref, atol=1e-5
    ), f"RadialBasis forward mismatch, max diff: {(out_xn - out_ref).abs().max().item()}"


def test_polynomial_envelope_matches_formula():
    """
    The polynomial cutoff envelope used across GemNet / GemNet-OC must implement the exact Klicpera-Groß-Günnemann formula.
    """
    from xanesnet.models.gemnet_oc.layers.radial_basis import PolynomialEnvelope

    p = 5
    env = PolynomialEnvelope(exponent=p)
    d_scaled = torch.linspace(0.0, 0.999, steps=11)
    out = env(d_scaled)
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    expected = 1 + a * d_scaled**p + b * d_scaled ** (p + 1) + c * d_scaled ** (p + 2)
    assert torch.allclose(out, expected, atol=1e-6)
    # Envelope must smoothly go to zero at the cutoff boundary.
    assert env(torch.tensor([1.0])).abs().item() < 1e-5
    # And is unity at the origin.
    assert torch.isclose(env(torch.tensor([0.0])), torch.tensor([1.0]))


def test_gemnet_bessel_basis_matches_formula():
    """
    GemNet's ``BesselBasisLayer`` (with its own polynomial envelope) must match the analytic formula for a fixed cutoff.
    """
    from xanesnet.models.gemnet.layers.basis import BesselBasisLayer

    layer = BesselBasisLayer(num_radial=4, cutoff=5.0, envelope_exponent=5)
    cutoff = 5.0
    d = torch.linspace(0.1, 4.9, steps=10)
    out = layer(d)
    d_scaled = d / cutoff
    p = 5
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    env = 1 + a * d_scaled**p + b * d_scaled ** (p + 1) + c * d_scaled ** (p + 2)
    freqs = layer.frequencies.detach()
    # GemNet's Bessel (non-OC) uses norm = sqrt(2/cutoff) and divides by d directly (NOT d_scaled), as in the source.
    norm = (2.0 / cutoff) ** 0.5
    bessel = norm * torch.sin(freqs * d_scaled[:, None]) / d[:, None]
    expected = env[:, None] * bessel
    assert torch.allclose(out, expected, atol=1e-5)


# 4. SPHERICAL / ANGLE HELPERS PARITY


def test_inner_product_clamped_matches_formula():
    """
    ``inner_product_clamped`` must equal the cosine of the angle between two unit-length vectors, clamped to [-1, 1].
    """
    from xanesnet.models.gemnet_oc.utils import inner_product_clamped

    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    a = a / a.norm(dim=-1, keepdim=True)
    b = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    b = b / b.norm(dim=-1, keepdim=True)
    out = inner_product_clamped(a, b)
    expected = torch.clamp((a * b).sum(dim=-1), min=-1.0, max=1.0)
    assert torch.allclose(out, expected, atol=1e-6)
    assert out.min().item() >= -1.0 and out.max().item() <= 1.0


def test_get_angle_matches_atan2():
    """
    ``get_angle(R1, R2)`` must equal ``atan2(||cross||, dot)`` on the input vector pairs.
    """
    from xanesnet.models.gemnet_oc.utils import get_angle

    R1 = torch.tensor([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    R2 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
    out = get_angle(R1, R2)
    cross = torch.linalg.cross(R1, R2)
    expected = torch.atan2(cross.norm(dim=-1), (R1 * R2).sum(dim=-1))
    assert torch.allclose(out, expected, atol=1e-6)
