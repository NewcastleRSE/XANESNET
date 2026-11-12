# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Unit tests for the ``radius``, ``cov_radius`` and ``voronoi`` graph builders."""

import pytest
import torch
from pymatgen.core import Lattice, Molecule, Structure

from xanesnet.graphs import GraphBuilder, GraphBuilderRegistry


def build(key: str, obj, cutoff: float, max_num_neighbors: int = 100, compute_vectors: bool = True, **extra):
    builder = GraphBuilderRegistry.create(
        key, graph_builder_type=key, cutoff=cutoff, max_num_neighbors=max_num_neighbors, **extra
    )
    return builder.build(obj, compute_vectors=compute_vectors)


def chain(*z_positions: float, species: str = "H") -> Molecule:
    return Molecule([species] * len(z_positions), [[0, 0, z] for z in z_positions])


def edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


@pytest.mark.parametrize(
    ("key", "extra"),
    [("radius", {}), ("cov_radius", {"cov_radii_scale": 1.2}), ("voronoi", {})],
)
def test_builders_are_registered_and_record_their_settings(key, extra):
    builder = GraphBuilderRegistry.create(key, graph_builder_type=key, cutoff=3.0, max_num_neighbors=7, **extra)

    assert isinstance(builder, GraphBuilder)
    assert (builder.graph_builder_type, builder.cutoff, builder.max_num_neighbors) == (key, 3.0, 7)


###############################################################################
#################################### RADIUS ###################################
###############################################################################


def test_radius_graph_connects_atoms_within_the_cutoff_in_both_directions():
    molecule = chain(0.0, 1.0, 3.0)

    edge_index, weight, vec, attr = build("radius", molecule, cutoff=1.5)

    assert edge_set(edge_index) == {(0, 1), (1, 0)}
    assert weight.tolist() == [1.0, 1.0]
    assert attr is None
    pairs = {edge: v for edge, v in zip(zip(*edge_index.tolist()), vec.tolist())}
    assert pairs[(0, 1)] == [0.0, 0.0, 1.0]
    assert pairs[(1, 0)] == [0.0, 0.0, -1.0]


def test_a_larger_cutoff_adds_the_longer_bonds():
    edge_index, weight, *_ = build("radius", chain(0.0, 1.0, 3.0), cutoff=5.0)

    assert edge_set(edge_index) == {(i, j) for i in range(3) for j in range(3) if i != j}
    assert sorted(set(weight.tolist())) == [1.0, 2.0, 3.0]


def test_edge_weights_are_the_lengths_of_the_edge_vectors():
    _, weight, vec, _ = build("radius", chain(0.0, 1.0, 3.0), cutoff=5.0)
    torch.testing.assert_close(weight, vec.norm(dim=-1))


def test_the_neighbour_limit_keeps_only_edges_that_survive_in_both_directions():
    # atom 2 would keep atom 1 (distance 2), but atom 1 prefers atom 0 (distance 1): the one-sided edge is removed
    edge_index, *_ = build("radius", chain(0.0, 1.0, 3.0), cutoff=5.0, max_num_neighbors=1)

    assert edge_set(edge_index) == {(0, 1), (1, 0)}


def test_vectors_can_be_skipped():
    _, weight, vec, _ = build("radius", chain(0.0, 1.0), cutoff=2.0, compute_vectors=False)

    assert vec is None
    assert weight.numel() == 2


def test_isolated_atoms_give_an_empty_graph():
    edge_index, weight, vec, _ = build("radius", chain(0.0, 10.0), cutoff=2.0)

    assert edge_index.shape == (2, 0)
    assert weight.numel() == 0 and vec.shape == (0, 3)


def test_periodic_structures_connect_atoms_to_their_periodic_images():
    structure = Structure(Lattice.cubic(2.0), ["Fe"], [[0, 0, 0]])

    edge_index, weight, vec, _ = build("radius", structure, cutoff=2.1)

    assert edge_index.shape == (2, 6)
    assert edge_set(edge_index) == {(0, 0)}
    torch.testing.assert_close(weight, torch.full((6,), 2.0))
    assert sorted(map(tuple, vec.tolist())) == sorted(
        [(2.0, 0, 0), (-2.0, 0, 0), (0, 2.0, 0), (0, -2.0, 0), (0, 0, 2.0), (0, 0, -2.0)]
    )


def test_periodic_structures_beyond_the_cutoff_give_an_empty_graph():
    structure = Structure(Lattice.cubic(5.0), ["Fe"], [[0, 0, 0]])
    edge_index, *_ = build("radius", structure, cutoff=2.0)
    assert edge_index.shape == (2, 0)


###############################################################################
################################# COV_RADIUS ##################################
###############################################################################


@pytest.mark.parametrize(("scale", "connected"), [(1.7, True), (1.5, False)])
def test_covalent_radius_graph_connects_atoms_closer_than_the_scaled_sum_of_covalent_radii(scale, connected):
    # hydrogen covalent radius 0.31 -> pair threshold 0.62 * scale, compared with a distance of 1.0
    edge_index, *_ = build("cov_radius", chain(0.0, 1.0), cutoff=5.0, cov_radii_scale=scale)

    assert (edge_set(edge_index) == {(0, 1), (1, 0)}) is connected
    assert (edge_index.shape[1] == 0) is not connected


def test_the_covalent_threshold_depends_on_the_elements():
    # Fe is much larger than H, so Fe-Fe at distance 2.0 is bonded while H-H at 2.0 is not
    fe_pair = build("cov_radius", chain(0.0, 2.0, species="Fe"), cutoff=5.0, cov_radii_scale=1.0)[0]
    h_pair = build("cov_radius", chain(0.0, 2.0, species="H"), cutoff=5.0, cov_radii_scale=1.0)[0]

    assert fe_pair.shape[1] == 2
    assert h_pair.shape[1] == 0


def test_the_cutoff_still_applies_on_top_of_the_covalent_threshold():
    edge_index, *_ = build("cov_radius", chain(0.0, 1.0), cutoff=0.5, cov_radii_scale=10.0)
    assert edge_index.shape[1] == 0


def test_covalent_radius_also_handles_periodic_structures():
    structure = Structure(Lattice.cubic(2.0), ["Fe"], [[0, 0, 0]])

    bonded = build("cov_radius", structure, cutoff=2.1, cov_radii_scale=1.0)[0]
    not_bonded = build("cov_radius", structure, cutoff=2.1, cov_radii_scale=0.5)[0]

    assert bonded.shape == (2, 6)
    assert not_bonded.shape == (2, 0)


###############################################################################
#################################### VORONOI ##################################
###############################################################################


def tetragonal() -> Structure:
    """One atom in a 3 x 3 x 4 box: facet areas 12 (normal to x and y) and 9 (normal to z)."""
    return Structure(Lattice.tetragonal(3.0, 4.0), ["Fe"], [[0, 0, 0]])


def test_voronoi_neighbours_are_the_atoms_sharing_a_facet_and_carry_the_facet_area():
    edge_index, weight, vec, area = build("voronoi", tetragonal(), cutoff=4.1)

    assert edge_index.shape == (2, 6)
    assert sorted(set(weight.tolist())) == pytest.approx([3.0, 4.0])
    torch.testing.assert_close(vec.norm(dim=-1), weight)
    by_length = {round(length, 3): sorted(set(area[weight.round(decimals=3) == length].tolist())) for length in (3.0, 4.0)}
    assert by_length[3.0] == pytest.approx([12.0])
    assert by_length[4.0] == pytest.approx([9.0])


def test_the_cutoff_limits_the_voronoi_neighbours():
    edge_index, weight, *_ = build("voronoi", tetragonal(), cutoff=3.5)

    assert edge_index.shape == (2, 4)
    assert weight.tolist() == pytest.approx([3.0] * 4)


@pytest.mark.parametrize(("threshold", "expected_edges"), [(10.0, 4), (8.0, 6), (13.0, 0)])
def test_small_facets_can_be_removed_by_an_absolute_area(threshold, expected_edges):
    edge_index, *_ = build("voronoi", tetragonal(), cutoff=4.1, min_facet_area=threshold)
    assert edge_index.shape[1] == expected_edges


@pytest.mark.parametrize(("percent", "expected_edges"), [("70%", 6), ("80%", 4), ("100%", 4)])
def test_small_facets_can_be_removed_relative_to_the_largest_facet(percent, expected_edges):
    edge_index, *_ = build("voronoi", tetragonal(), cutoff=4.1, min_facet_area=percent)
    assert edge_index.shape[1] == expected_edges


def test_voronoi_graphs_are_symmetric_and_can_skip_vectors():
    edge_index, _, _, area = build("voronoi", tetragonal(), cutoff=4.1)
    edges = edge_set(edge_index)
    assert {(dst, src) for src, dst in edges} == edges
    assert area.shape == (6,)

    _, _, no_vec, _ = build("voronoi", tetragonal(), cutoff=4.1, compute_vectors=False)
    assert no_vec is None


def test_voronoi_of_molecules_without_bounded_facets_is_empty():
    # a lone tetrahedron has no bounded Voronoi facets, and fewer than four atoms cannot be tessellated
    tetrahedron = Molecule(["H"] * 4, [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

    assert build("voronoi", tetrahedron, cutoff=5.0)[0].shape == (2, 0)
    assert build("voronoi", chain(0.0, 1.0), cutoff=5.0)[0].shape == (2, 0)
