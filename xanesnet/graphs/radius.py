# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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

"""Radius-graph and covalent-radius edge construction for XANESNET."""

import numpy as np
import torch
from ase.data import covalent_radii
from pymatgen.core import Molecule, Structure
from torch_geometric.nn import radius_graph

from .base import GraphBuilder
from .registry import GraphBuilderRegistry
from .utils.symmetrize import symmetrize_directed_edges, truncate_per_source


def _pair_cov_cutoff(z_src: np.ndarray, z_dst: np.ndarray, cov_radii_scale: float) -> np.ndarray:
    """Compute per-pair covalent-radius cutoffs.

    Returns ``cov_radii_scale * (r_cov_src + r_cov_dst)`` in **angstroms**
    using ASE's covalent radii table (Cordero et al.).

    Args:
        z_src: ``(E,)`` int -- atomic numbers of source atoms.
        z_dst: ``(E,)`` int -- atomic numbers of destination atoms.
        cov_radii_scale: Multiplicative scale factor applied to the sum of
            covalent radii.

    Returns:
        ``(E,)`` float64 -- per-pair distance threshold in **angstroms**.
    """
    cr = np.asarray(covalent_radii, dtype=np.float64)
    return cov_radii_scale * (cr[z_src] + cr[z_dst])


def _edges_from_structure(
    structure: Structure,
    cutoff: float,
    max_num_neighbors: int,
    cov_radii_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build edges for a periodic structure using pymatgen's neighbor search.

    Correctly handles cutoffs larger than the unit cell by expanding images.
    Truncation is performed after the full neighbor list is built; edges are
    then symmetrised so the returned graph is guaranteed to be bidirectional.

    Args:
        structure: Periodic pymatgen ``Structure``.
        cutoff: Maximum edge length in **angstroms**.
        max_num_neighbors: Maximum outgoing edges retained per source node
            (shortest first).
        cov_radii_scale: If given, edges with
            ``d > cov_radii_scale * (r_cov_src + r_cov_dst)`` are additionally
            dropped (``cutoff`` still acts as a hard maximum).

    Returns:
        A 3-tuple ``(edge_index, edge_weight, edge_vec)``:

        - ``edge_index``: ``(2, E)`` int64.
        - ``edge_weight``: ``(E,)`` float32 -- edge lengths in **angstroms**.
        - ``edge_vec``: ``(E, 3)`` float32 -- displacement vectors
          ``pos[dst] - pos[src]``.
    """
    all_neighbors = structure.get_all_neighbors(r=cutoff)
    z_all = np.asarray(structure.atomic_numbers, dtype=np.int64)
    src: list[int] = []
    dst: list[int] = []
    dists: list[float] = []
    edge_vectors: list[np.ndarray] = []

    for i, site_neighbors in enumerate(all_neighbors):
        for neighbor in site_neighbors:
            src.append(i)
            dst.append(neighbor.index)
            dists.append(neighbor.nn_distance)
            edge_vectors.append(neighbor.coords - structure.cart_coords[i])

    if len(src) == 0:
        return (
            torch.zeros(2, 0, dtype=torch.int64),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, 3, dtype=torch.float32),
        )

    src_np = np.asarray(src, dtype=np.int64)
    dst_np = np.asarray(dst, dtype=np.int64)
    dists_np = np.asarray(dists, dtype=np.float64)
    vecs_np = np.asarray(edge_vectors, dtype=np.float64).reshape(-1, 3)

    if cov_radii_scale is not None:
        pair_cut = _pair_cov_cutoff(z_all[src_np], z_all[dst_np], cov_radii_scale)
        keep = dists_np <= pair_cut
        src_np = src_np[keep]
        dst_np = dst_np[keep]
        dists_np = dists_np[keep]
        vecs_np = vecs_np[keep]

    if src_np.shape[0] == 0:
        return (
            torch.zeros(2, 0, dtype=torch.int64),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, 3, dtype=torch.float32),
        )

    edge_index = torch.tensor(np.stack([src_np, dst_np], axis=0), dtype=torch.int64)
    edge_weight = torch.tensor(dists_np, dtype=torch.float32)
    edge_vec = torch.tensor(vecs_np, dtype=torch.float32).reshape(-1, 3)

    edge_index, edge_weight, edge_vec, _ = truncate_per_source(
        edge_index, edge_weight, edge_vec, None, max_num_neighbors
    )
    edge_index, edge_weight, edge_vec, _ = symmetrize_directed_edges(edge_index, edge_weight, edge_vec, None)
    return edge_index, edge_weight, edge_vec


def _edges_from_molecule(
    molecule: Molecule,
    cutoff: float,
    max_num_neighbors: int,
    cov_radii_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build edges for a non-periodic molecule using PyG's ``radius_graph``.

    Bidirectionality is enforced after ``max_num_neighbors`` truncation.

    Args:
        molecule: Non-periodic pymatgen ``Molecule``.
        cutoff: Maximum edge length in **angstroms**.
        max_num_neighbors: Maximum outgoing edges retained per source node
            (shortest first).
        cov_radii_scale: If given, edges with
            ``d > cov_radii_scale * (r_cov_src + r_cov_dst)`` are additionally
            dropped (``cutoff`` still acts as a hard maximum).

    Returns:
        A 3-tuple ``(edge_index, edge_weight, edge_vec)``:

        - ``edge_index``: ``(2, E)`` int64.
        - ``edge_weight``: ``(E,)`` float32 -- edge lengths in **angstroms**.
        - ``edge_vec``: ``(E, 3)`` float32 -- displacement vectors
          ``pos[dst] - pos[src]``.
    """
    pos = torch.tensor(molecule.cart_coords, dtype=torch.float32)
    # Use a generous max here; we do our own truncation below so we can
    # symmetrise after truncation.
    edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=pos.shape[0])
    row, col = edge_index
    edge_vec = pos[col] - pos[row]
    edge_weight = edge_vec.norm(dim=-1)

    if cov_radii_scale is not None and edge_index.shape[1] > 0:
        z_all = np.asarray(molecule.atomic_numbers, dtype=np.int64)
        pair_cut = _pair_cov_cutoff(z_all[row.numpy()], z_all[col.numpy()], cov_radii_scale)
        keep = edge_weight.numpy() <= pair_cut
        keep_t = torch.from_numpy(keep)
        edge_index = edge_index[:, keep_t]
        edge_vec = edge_vec[keep_t]
        edge_weight = edge_weight[keep_t]

    edge_index, edge_weight, edge_vec, _ = truncate_per_source(
        edge_index, edge_weight, edge_vec, None, max_num_neighbors
    )
    edge_index, edge_weight, edge_vec, _ = symmetrize_directed_edges(edge_index, edge_weight, edge_vec, None)
    return edge_index, edge_weight, edge_vec


@GraphBuilderRegistry.register("radius")
class RadiusGraphBuilder(GraphBuilder):
    """Radius-graph builder.

    Two atoms are connected iff their Cartesian distance is at most
    ``cutoff``. For periodic structures this uses pymatgen's PBC-aware
    neighbor search, which correctly handles cutoffs larger than the unit
    cell by expanding images. For non-periodic molecules this uses PyG's
    ``radius_graph``.

    Args:
        graph_builder_type: Registered builder identifier (``"radius"``).
        cutoff: Maximum edge length in **angstroms**.
        max_num_neighbors: Maximum outgoing edges retained per source node
            (shortest first). Bidirectionality is enforced afterwards.
    """

    def __init__(
        self,
        graph_builder_type: str,
        cutoff: float,
        max_num_neighbors: int,
    ) -> None:
        """Initialize ``RadiusGraphBuilder``."""
        super().__init__(graph_builder_type, cutoff, max_num_neighbors)

    def build(
        self,
        pmg_obj: Structure | Molecule,
        compute_vectors: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Build a radius graph for ``pmg_obj``.

        Args:
            pmg_obj: Periodic ``Structure`` or non-periodic ``Molecule``.
            compute_vectors: If ``False``, ``edge_vec`` is returned as
                ``None`` in the output tuple.

        Returns:
            ``(edge_index, edge_weight, edge_vec, edge_attr)`` where
            ``edge_attr`` is always ``None`` for the radius method.
        """
        if isinstance(pmg_obj, Structure):
            edge_index, edge_weight, edge_vec = _edges_from_structure(pmg_obj, self.cutoff, self.max_num_neighbors)
        else:
            edge_index, edge_weight, edge_vec = _edges_from_molecule(pmg_obj, self.cutoff, self.max_num_neighbors)
        if not compute_vectors:
            edge_vec = None
        return edge_index, edge_weight, edge_vec, None


@GraphBuilderRegistry.register("cov_radius")
class CovRadiusGraphBuilder(GraphBuilder):
    """Covalent-radius edge construction.

    An edge ``(i, j)`` is kept iff
    ``d(i, j) <= min(cutoff, cov_radii_scale * (r_cov_i + r_cov_j))``,
    where ``r_cov`` values are taken from ASE's Cordero covalent-radii
    table. ``cutoff`` acts as a hard maximum (useful for speed and for
    capping extreme element combinations).

    Args:
        graph_builder_type: Registered builder identifier (``"cov_radius"``).
        cutoff: Hard maximum edge length in **angstroms**.
        max_num_neighbors: Maximum outgoing edges retained per source node
            (shortest first).
        cov_radii_scale: Scale factor applied to the sum of covalent radii.
    """

    def __init__(
        self,
        graph_builder_type: str,
        cutoff: float,
        max_num_neighbors: int,
        cov_radii_scale: float,
    ) -> None:
        """Initialize ``CovRadiusGraphBuilder``."""
        super().__init__(graph_builder_type, cutoff, max_num_neighbors)
        self.cov_radii_scale = cov_radii_scale

    def build(
        self,
        pmg_obj: Structure | Molecule,
        compute_vectors: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Build a covalent-radius graph for ``pmg_obj``.

        Args:
            pmg_obj: Periodic ``Structure`` or non-periodic ``Molecule``.
            compute_vectors: If ``False``, ``edge_vec`` is returned as
                ``None`` in the output tuple.

        Returns:
            ``(edge_index, edge_weight, edge_vec, edge_attr)`` where
            ``edge_attr`` is always ``None`` for the covalent-radius method.
        """
        if isinstance(pmg_obj, Structure):
            edge_index, edge_weight, edge_vec = _edges_from_structure(
                pmg_obj, self.cutoff, self.max_num_neighbors, cov_radii_scale=self.cov_radii_scale
            )
        else:
            edge_index, edge_weight, edge_vec = _edges_from_molecule(
                pmg_obj, self.cutoff, self.max_num_neighbors, cov_radii_scale=self.cov_radii_scale
            )
        if not compute_vectors:
            edge_vec = None
        return edge_index, edge_weight, edge_vec, None
