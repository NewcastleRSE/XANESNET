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


def edges_from_structure(
    structure: Structure,
    cutoff: float,
    max_num_neighbors: int,
    compute_vectors: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Build edge_index, edge_weight (and optionally displacement vectors) for a
    periodic structure. Uses pymatgen's minimum-image neighbor search, which
    correctly handles cutoffs larger than the unit cell by expanding images.
    """
    all_neighbors = structure.get_all_neighbors(r=cutoff)
    src: list[int] = []
    dst: list[int] = []
    dists: list[float] = []
    edge_vectors: list[np.ndarray] | None = [] if compute_vectors else None

    for i, site_neighbors in enumerate(all_neighbors):
        sorted_neighbors = sorted(site_neighbors, key=lambda n: n.nn_distance)
        for neighbor in sorted_neighbors[:max_num_neighbors]:
            src.append(i)
            dst.append(neighbor.index)
            dists.append(neighbor.nn_distance)
            if edge_vectors is not None:
                edge_vectors.append(neighbor.coords - structure.cart_coords[i])

    edge_index = torch.tensor([src, dst], dtype=torch.int64)
    edge_weight = torch.tensor(dists, dtype=torch.float32)
    edge_vec = (
        torch.tensor(np.array(edge_vectors), dtype=torch.float32).reshape(-1, 3) if edge_vectors is not None else None
    )
    return edge_index, edge_weight, edge_vec


def edges_from_molecule(
    molecule: Molecule,
    cutoff: float,
    max_num_neighbors: int,
    compute_vectors: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Build edge_index, edge_weight (and optionally displacement vectors) for a
    non-periodic molecule.
    """
    pos = torch.tensor(molecule.cart_coords, dtype=torch.float32)
    edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=max_num_neighbors)
    row, col = edge_index
    edge_weight = (pos[row] - pos[col]).norm(dim=-1)
    # displacement from source to target (row -> col)
    edge_vec = pos[col] - pos[row] if compute_vectors else None
    return edge_index, edge_weight, edge_vec


def build_edges(
    pmg_obj: Structure | Molecule,
    cutoff: float,
    max_num_neighbors: int,
    compute_vectors: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Dispatch edge construction by pymatgen type. Supports periodic Structure
    and non-periodic Molecule uniformly.
    """
    if isinstance(pmg_obj, Structure):
        return edges_from_structure(pmg_obj, cutoff, max_num_neighbors, compute_vectors)
    return edges_from_molecule(pmg_obj, cutoff, max_num_neighbors, compute_vectors)
