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
from pymatgen.core import Molecule, Structure
from torch_geometric.nn import radius_graph

from .symmetrize import symmetrize_directed_edges, truncate_per_source


def edges_from_structure(
    structure: Structure,
    cutoff: float,
    max_num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build edge_index, edge_weight and displacement vectors for a periodic
    structure using pymatgen's minimum-image neighbour search, which correctly
    handles cutoffs larger than the unit cell by expanding images.

    Truncation is performed AFTER the full neighbour list is built, and edges
    are symmetrised so the returned graph is guaranteed to be bidirectional.
    """
    all_neighbors = structure.get_all_neighbors(r=cutoff)
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

    edge_index = torch.tensor([src, dst], dtype=torch.int64)
    edge_weight = torch.tensor(dists, dtype=torch.float32)
    edge_vec = torch.tensor(np.array(edge_vectors), dtype=torch.float32).reshape(-1, 3)

    edge_index, edge_weight, edge_vec, _ = truncate_per_source(
        edge_index, edge_weight, edge_vec, None, max_num_neighbors
    )
    edge_index, edge_weight, edge_vec, _ = symmetrize_directed_edges(edge_index, edge_weight, edge_vec, None)
    return edge_index, edge_weight, edge_vec


def edges_from_molecule(
    molecule: Molecule,
    cutoff: float,
    max_num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build edge_index, edge_weight and displacement vectors for a non-periodic
    molecule. Uses PyG's ``radius_graph`` (which yields both directions for
    every pair within radius by default) then enforces bidirectionality after
    ``max_num_neighbors`` truncation.
    """
    pos = torch.tensor(molecule.cart_coords, dtype=torch.float32)
    # Use a generous max here; we do our own truncation below so we can
    # symmetrise after truncation.
    edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=pos.shape[0])
    row, col = edge_index
    edge_vec = pos[col] - pos[row]
    edge_weight = edge_vec.norm(dim=-1)

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
