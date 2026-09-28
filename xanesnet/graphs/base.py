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

"""Abstract base class for all XANESNET graph builders."""

from abc import ABC, abstractmethod

import torch
from pymatgen.core import Molecule, Structure


class GraphBuilder(ABC):
    """Abstract base class for all XANESNET graph builders.

    A graph builder converts a pymatgen ``Structure`` or ``Molecule`` into a
    directed edge list with per-edge metadata used by graph neural networks.
    Concrete subclasses implement one edge-construction strategy each and
    register themselves with :data:`GraphBuilderRegistry`.

    All builders return bidirectional graphs: every emitted edge ``(i -> j)``
    has a matching reverse edge ``(j -> i)``, so downstream models can treat
    the returned ``edge_index`` as an undirected graph.

    Args:
        graph_builder_type: Identifier string for the concrete builder type.
        cutoff: Maximum edge length in **angstroms**. The precise meaning is
            method-dependent (hard cutoff for radius/voronoi; hard upper
            bound alongside a covalent-radius test for ``cov_radius``).
        max_num_neighbors: Maximum outgoing edges retained per source node
            (shortest first). Bidirectionality is enforced after this
            truncation.
    """

    def __init__(
        self,
        graph_builder_type: str,
        cutoff: float,
        max_num_neighbors: int,
    ) -> None:
        """Initialize ``GraphBuilder``."""
        self.graph_builder_type = graph_builder_type
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    @abstractmethod
    def build(
        self,
        pmg_obj: Structure | Molecule,
        compute_vectors: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Build a graph from a pymatgen ``Structure`` or ``Molecule``.

        Supports periodic ``Structure`` and non-periodic ``Molecule`` objects
        uniformly.

        Args:
            pmg_obj: The pymatgen object whose atoms are graph nodes.
            compute_vectors: If ``False``, ``edge_vec`` is returned as
                ``None`` to save memory in downstream consumers that do not
                need it. Geometry-aware methods still compute vectors
                internally (they are needed for symmetrisation).

        Returns:
            A 4-tuple ``(edge_index, edge_weight, edge_vec, edge_attr)``:

            - ``edge_index``: ``(2, E)`` int64 -- source/destination node
              indices.
            - ``edge_weight``: ``(E,)`` float32 -- edge length in
              **angstroms**.
            - ``edge_vec``: ``(E, 3)`` float32 or ``None`` -- displacement
              vectors ``pos[dst] - pos[src]``.
            - ``edge_attr``: ``(E,)`` float32 or ``None`` -- optional
              per-edge scalar (facet area for Voronoi, ``None`` otherwise).
        """
        ...
