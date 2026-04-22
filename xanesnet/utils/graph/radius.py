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

import numpy as np
import torch
from ase.data import covalent_radii
from pymatgen.core import Molecule, Structure
from torch_geometric.nn import radius_graph

from .symmetrize import symmetrize_directed_edges, truncate_per_source


def _pair_cov_cutoff(z_src: np.ndarray, z_dst: np.ndarray, cov_radii_scale: float) -> np.ndarray:
    """
    Per-pair covalent-radius cutoff ``cov_radii_scale * (r_cov_src + r_cov_dst)``
    in Å, using ASE's covalent radii table (Cordero et al.).
    """
    cr = np.asarray(covalent_radii, dtype=np.float64)
    return cov_radii_scale * (cr[z_src] + cr[z_dst])


def edges_from_structure(
    structure: Structure,
    cutoff: float,
    max_num_neighbors: int,
    cov_radii_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build edge_index, edge_weight and displacement vectors for a periodic
    structure using pymatgen's minimum-image neighbour search, which correctly
    handles cutoffs larger than the unit cell by expanding images.

    Truncation is performed AFTER the full neighbour list is built, and edges
    are symmetrised so the returned graph is guaranteed to be bidirectional.

    If ``cov_radii_scale`` is given, edges with
    ``d > cov_radii_scale * (r_cov_src + r_cov_dst)`` are additionally
    dropped (``cutoff`` still acts as a hard maximum).
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


def edges_from_molecule(
    molecule: Molecule,
    cutoff: float,
    max_num_neighbors: int,
    cov_radii_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build edge_index, edge_weight and displacement vectors for a non-periodic
    molecule. Uses PyG's ``radius_graph`` (which yields both directions for
    every pair within radius by default) then enforces bidirectionality after
    ``max_num_neighbors`` truncation.

    If ``cov_radii_scale`` is given, edges with
    ``d > cov_radii_scale * (r_cov_src + r_cov_dst)`` are additionally
    dropped (``cutoff`` still acts as a hard maximum).
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


def build_edges_radius(
    pmg_obj: Structure | Molecule,
    cutoff: float,
    max_num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Radius-graph edge construction. Returns
    ``(edge_index, edge_weight, edge_vec, edge_attr)`` where ``edge_attr`` is
    always ``None`` for the radius method.
    """
    if isinstance(pmg_obj, Structure):
        edge_index, edge_weight, edge_vec = edges_from_structure(pmg_obj, cutoff, max_num_neighbors)
    else:
        edge_index, edge_weight, edge_vec = edges_from_molecule(pmg_obj, cutoff, max_num_neighbors)
    return edge_index, edge_weight, edge_vec, None


def build_edges_cov_radius(
    pmg_obj: Structure | Molecule,
    cutoff: float,
    max_num_neighbors: int,
    cov_radii_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Covalent-radius edge construction. An edge (i, j) is kept iff

        d(i, j) <= min(cutoff, cov_radii_scale * (r_cov_i + r_cov_j))

    with r_cov taken from ASE's Cordero covalent-radii table. ``cutoff``
    acts as a hard maximum (useful for speed and for capping extreme element
    combinations). Returns ``(edge_index, edge_weight, edge_vec, None)``.
    """
    if isinstance(pmg_obj, Structure):
        edge_index, edge_weight, edge_vec = edges_from_structure(
            pmg_obj, cutoff, max_num_neighbors, cov_radii_scale=cov_radii_scale
        )
    else:
        edge_index, edge_weight, edge_vec = edges_from_molecule(
            pmg_obj, cutoff, max_num_neighbors, cov_radii_scale=cov_radii_scale
        )
    return edge_index, edge_weight, edge_vec, None
