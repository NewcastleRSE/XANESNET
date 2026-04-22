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
from pymatgen.core import Molecule, Structure

from .radius import build_edges_radius
from .voronoi import build_edges_voronoi

GRAPH_METHODS = ("radius", "voronoi")


def build_edges(
    pmg_obj: Structure | Molecule,
    cutoff: float,
    max_num_neighbors: int,
    compute_vectors: bool = True,
    method: str = "radius",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Dispatch edge construction by ``method``. Supports periodic ``Structure``
    and non-periodic ``Molecule`` uniformly.

    Parameters
    ----------
    pmg_obj:
        The pymatgen object whose atoms are graph nodes.
    cutoff:
        Maximum edge length in angstroms. For the Voronoi method this bounds
        the edge length after the tessellation.
    max_num_neighbors:
        Maximum outgoing edges kept per source node (shortest first). The
        returned graph is guaranteed to be bidirectional regardless of this
        truncation.
    compute_vectors:
        If False, ``edge_vec`` is returned as ``None`` to save memory in
        downstream consumers that do not need it. Geometry-aware methods still
        compute vectors internally (they are needed for symmetrisation).
    method:
        ``"radius"`` (default) or ``"voronoi"``.

    Returns
    -------
    edge_index: ``[2, E]`` long
    edge_weight: ``[E]`` float -- edge length in angstroms
    edge_vec: ``[E, 3]`` float or ``None``
    edge_attr: ``[E]`` float or ``None`` -- extra per-edge scalar
        (facet area for the Voronoi method, ``None`` for the radius method)
    """
    method = method.lower()
    if method == "radius":
        edge_index, edge_weight, edge_vec, edge_attr = build_edges_radius(
            pmg_obj, cutoff, max_num_neighbors
        )
    elif method == "voronoi":
        edge_index, edge_weight, edge_vec, edge_attr = build_edges_voronoi(
            pmg_obj, cutoff, max_num_neighbors
        )
    else:
        raise ValueError(
            f"Unknown graph method {method!r}; expected one of {GRAPH_METHODS}"
        )

    if not compute_vectors:
        edge_vec = None
    return edge_index, edge_weight, edge_vec, edge_attr
