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

"""Unit tests for the edge-list utilities shared by the graph builders."""

import math

import pytest
import torch
from pymatgen.core import Lattice, Molecule, Structure

from xanesnet.graphs.utils import (
    build_target_site_paths,
    compute_id_swap,
    compute_triplets,
    compute_triplets_and_angles,
    symmetrize_directed_edges,
    truncate_per_source,
)


def vectors(*rows: tuple[float, float, float]) -> torch.Tensor:
    """Build a float32 tensor from three-dimensional vector rows.

    Args:
        rows: Vector components to stack into a tensor.

    Returns:
        Tensor containing the supplied vectors.
    """
    return torch.tensor(rows, dtype=torch.float32)


def star_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """Node 0 connected in both directions to nodes 1, 2, 3 along the axes."""
    leaves = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    src = [0, 0, 0, 1, 2, 3]
    dst = [1, 2, 3, 0, 0, 0]
    vec = [*leaves, *[(-x, -y, -z) for x, y, z in leaves]]
    return torch.tensor([src, dst]), vectors(*vec)


def right_angle_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """Node 0 at the corner, nodes 1 and 2 along x and y."""
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]])
    return edge_index, vectors((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))


###############################################################################
################################ TRUNCATION ###################################
###############################################################################


def test_truncation_keeps_the_nearest_neighbours_of_every_source():
    edge_index = torch.tensor([[0, 0, 0, 1], [1, 2, 3, 0]])
    weight = torch.tensor([3.0, 1.0, 2.0, 5.0])
    vec = torch.arange(12.0).reshape(4, 3)
    attr = torch.tensor([10.0, 20.0, 30.0, 40.0])

    index, kept_weight, kept_vec, kept_attr = truncate_per_source(edge_index, weight, vec, attr, 2)

    assert kept_attr is not None
    assert index.tolist() == [[0, 0, 1], [2, 3, 0]]
    assert kept_weight.tolist() == [1.0, 2.0, 5.0]
    assert kept_vec.tolist() == vec[[1, 2, 3]].tolist()
    assert kept_attr.tolist() == [20.0, 30.0, 40.0]


@pytest.mark.parametrize("limit", [0, -1, None])
def test_truncation_is_disabled_by_a_non_positive_limit(limit):
    edge_index = torch.tensor([[0, 0], [1, 2]])
    weight = torch.tensor([1.0, 2.0])

    index, *_ = truncate_per_source(edge_index, weight, torch.zeros(2, 3), None, limit)

    assert index is edge_index


def test_truncation_of_an_empty_graph_is_a_no_op():
    index, weight, vec, attr = truncate_per_source(
        torch.zeros(2, 0, dtype=torch.int64), torch.zeros(0), torch.zeros(0, 3), None, 3
    )
    assert index.shape == (2, 0) and attr is None


###############################################################################
################################ SYMMETRISATION ###############################
###############################################################################


def test_edges_without_a_reverse_partner_are_dropped():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 2]])
    vec = vectors((1, 0, 0), (-1, 0, 0), (0, 1, 0))
    weight = torch.tensor([1.0, 1.0, 1.0])
    attr = torch.tensor([5.0, 6.0, 7.0])

    index, _, kept_vec, kept_attr = symmetrize_directed_edges(edge_index, weight, vec, attr)

    assert kept_attr is not None
    assert index.tolist() == [[0, 1], [1, 0]]
    assert kept_vec.tolist() == vec[:2].tolist()
    assert kept_attr.tolist() == [5.0, 6.0]


def test_a_partner_must_be_the_exact_reverse_displacement_not_just_the_reverse_atom_pair():
    # 0 -> 0 through two different periodic images: only mirrored displacements are partners
    edge_index = torch.tensor([[0, 0, 0], [0, 0, 0]])
    vec = vectors((2, 0, 0), (-2, 0, 0), (0, 2, 0))

    index, _, kept_vec, _ = symmetrize_directed_edges(edge_index, torch.ones(3), vec, None)

    assert kept_vec.tolist() == [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
    assert index.shape == (2, 2)


def test_duplicated_edges_are_kept_only_as_often_as_their_reverse_exists():
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 0]])
    vec = vectors((1, 0, 0), (1, 0, 0), (-1, 0, 0))

    index, *_ = symmetrize_directed_edges(edge_index, torch.ones(3), vec, None)

    assert index.tolist() == [[0, 1], [1, 0]]


def test_symmetrising_a_symmetric_graph_changes_nothing_and_an_empty_graph_is_accepted():
    edge_index, vec = star_graph()
    index, *_ = symmetrize_directed_edges(edge_index, torch.ones(6), vec, None)
    assert index.tolist() == edge_index.tolist()

    empty, *_ = symmetrize_directed_edges(torch.zeros(2, 0, dtype=torch.int64), torch.zeros(0), torch.zeros(0, 3), None)
    assert empty.shape == (2, 0)


###############################################################################
################################ DIRECTIONAL INDICES ##########################
###############################################################################


def test_id_swap_points_every_edge_to_its_reverse():
    edge_index, vec = star_graph()

    id_swap = compute_id_swap(edge_index, vec)

    assert id_swap.tolist() == [3, 4, 5, 0, 1, 2]
    assert torch.equal(edge_index[:, id_swap], edge_index.flip(0))
    torch.testing.assert_close(vec[id_swap], -vec)


def test_id_swap_of_a_periodic_self_edge_pairs_opposite_images():
    edge_index = torch.tensor([[0, 0], [0, 0]])

    assert compute_id_swap(edge_index, vectors((2, 0, 0), (-2, 0, 0))).tolist() == [1, 0]


def test_id_swap_requires_a_symmetric_graph():
    with pytest.raises(ValueError, match="no matching counter-edge"):
        compute_id_swap(torch.tensor([[0], [1]]), vectors((1, 0, 0)))

    with pytest.raises(ValueError, match="multiplicities"):
        compute_id_swap(torch.tensor([[0, 0, 1], [1, 1, 0]]), vectors((1, 0, 0), (1, 0, 0), (-1, 0, 0)))


def test_id_swap_of_an_empty_graph_is_empty():
    assert compute_id_swap(torch.zeros(2, 0, dtype=torch.int64), torch.zeros(0, 3)).numel() == 0


def test_triplets_pair_every_edge_with_the_other_edges_entering_the_same_atom():
    edge_index, _ = star_graph()

    reduce_ca, expand_ba, kidx = compute_triplets(edge_index, num_nodes=4)

    # three edges enter the centre, each combines with the two others
    assert reduce_ca.numel() == 6
    assert reduce_ca.tolist() == sorted(reduce_ca.tolist())
    assert (reduce_ca != expand_ba).all()
    assert torch.equal(edge_index[1][reduce_ca], edge_index[1][expand_ba])
    assert set(edge_index[1][reduce_ca].tolist()) == {0}
    assert kidx.tolist() == [0, 1] * 3


def test_triplets_of_an_empty_graph_are_empty():
    for indices in compute_triplets(torch.zeros(2, 0, dtype=torch.int64), num_nodes=3):
        assert indices.numel() == 0


@pytest.mark.parametrize("periodic", [False, True])
def test_triplet_angles_are_measured_at_the_central_atom(periodic):
    edge_index, vec = right_angle_graph()

    angle, idx_kj, idx_ji = compute_triplets_and_angles(edge_index, vec, num_nodes=3, is_periodic=periodic)

    assert angle.numel() == 2  # k -> j -> i around the corner atom, in both orders
    torch.testing.assert_close(angle, torch.full((2,), math.pi / 2))
    # the first leg ends where the second begins, and the path does not return to its start
    assert torch.equal(edge_index[1][idx_kj], edge_index[0][idx_ji])
    assert (edge_index[0][idx_kj] != edge_index[1][idx_ji]).all()


def test_collinear_triplets_have_angle_pi():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    vec = vectors((1, 0, 0), (-1, 0, 0), (1, 0, 0), (-1, 0, 0))

    angle, *_ = compute_triplets_and_angles(edge_index, vec, num_nodes=3, is_periodic=False)

    torch.testing.assert_close(angle, torch.full((2,), math.pi))


def test_periodic_triplets_drop_paths_that_bounce_straight_back():
    # one atom with periodic images along +x and -x: the path -x then +x returns to the start
    edge_index = torch.tensor([[0, 0], [0, 0]])
    vec = vectors((2, 0, 0), (-2, 0, 0))

    angle, *_ = compute_triplets_and_angles(edge_index, vec, num_nodes=1, is_periodic=True)

    assert angle.numel() == 0


###############################################################################
################################ TARGET-SITE PATHS ############################
###############################################################################


def test_paths_connect_pairs_of_neighbours_of_the_target_site_nearest_first():
    molecule = Molecule(["Fe", "O", "O", "O"], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-2, 0, 0]])

    paths = build_target_site_paths(molecule, target_site_idx=0, cutoff=2.5, max_paths=10)

    # pairs ordered by r0j + r0k + 0.5 * rjk
    assert list(zip(paths["path_j"].tolist(), paths["path_k"].tolist())) == [(1, 2), (2, 3), (1, 3)]
    torch.testing.assert_close(paths["path_r0j"], torch.tensor([1.0, 1.0, 1.0]))
    torch.testing.assert_close(paths["path_r0k"], torch.tensor([1.0, 2.0, 2.0]))
    torch.testing.assert_close(paths["path_rjk"], torch.tensor([2**0.5, 5**0.5, 3.0]))
    torch.testing.assert_close(paths["path_cosangle"], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6, rtol=0)


def test_paths_are_limited_to_the_best_scoring_pairs():
    molecule = Molecule(["Fe", "O", "O", "O"], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-2, 0, 0]])

    paths = build_target_site_paths(molecule, 0, cutoff=2.5, max_paths=2)

    assert list(zip(paths["path_j"].tolist(), paths["path_k"].tolist())) == [(1, 2), (2, 3)]


@pytest.mark.parametrize("cutoff", [0.5, 1.5])
def test_fewer_than_two_neighbours_give_empty_paths(cutoff):
    molecule = Molecule(["Fe", "O", "O"], [[0, 0, 0], [1, 0, 0], [5, 0, 0]])

    paths = build_target_site_paths(molecule, 0, cutoff=cutoff, max_paths=10)

    assert set(paths) == {"path_j", "path_k", "path_r0j", "path_r0k", "path_rjk", "path_cosangle"}
    assert all(values.numel() == 0 for values in paths.values())


def test_periodic_paths_run_through_images_of_the_neighbours_but_not_the_site_itself():
    structure = Structure(Lattice.cubic(2.0), ["Fe"], [[0, 0, 0]])

    paths = build_target_site_paths(structure, 0, cutoff=2.1, max_paths=100)

    assert paths["path_j"].numel() == 15  # pairs of the six neighbouring images
    torch.testing.assert_close(paths["path_r0j"], torch.full((15,), 2.0))
    assert paths["path_cosangle"].min().item() == pytest.approx(-1.0)
