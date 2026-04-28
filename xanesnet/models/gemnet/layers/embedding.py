# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Atom and edge embedding modules for GemNet."""

import numpy as np
import torch

from .base import Dense


class AtomEmbedding(torch.nn.Module):
    """Initial atom embeddings based on atomic number.

    Args:
        emb_size: Atom embedding dimension.
        num_elements: Number of distinct element types in the embedding table.
            The default of 94 supports atomic numbers 1-94 (up to Pu).
            Internally uses ``z - 1`` indexing.
    """

    def __init__(self, emb_size: int, num_elements: int = 94) -> None:
        """Initialize ``AtomEmbedding``."""
        super().__init__()
        self.emb_size = emb_size
        self.num_elements = num_elements
        self.embeddings = torch.nn.Embedding(num_elements, emb_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialize the embedding table with uniform distribution in ``[-sqrt(3), sqrt(3)]``."""
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Look up atom embeddings by atomic number.

        Args:
            z: Atomic numbers, shape ``(nAtoms,)``.

        Returns:
            Atom embeddings of shape ``(nAtoms, emb_size)``.
        """
        h = self.embeddings(z - 1)  # -1 because z.min() == 1 (Hydrogen)
        return h


class EdgeEmbedding(torch.nn.Module):
    """Edge embeddings from atom-embedding concatenation followed by a dense layer.

    Concatenates the embeddings of the source and target atoms with the
    edge radial basis features, then applies a :class:`Dense` projection.

    Args:
        atom_features: Atom embedding dimension.
        edge_features: Edge radial basis feature dimension.
        out_features: Output edge embedding dimension.
        activation: Activation function name.
    """

    def __init__(
        self,
        atom_features: int,
        edge_features: int,
        out_features: int,
        activation: str,
    ) -> None:
        """Initialize ``EdgeEmbedding``."""
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def reset_parameters(self) -> None:
        """Re-initialize the inner dense layer."""
        self.dense.reset_parameters()

    def forward(
        self,
        h: torch.Tensor,
        m_rbf: torch.Tensor,
        idnb_a: torch.Tensor,
        idnb_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge embeddings from atom embeddings and edge features.

        Args:
            h: Atom embeddings, shape ``(nAtoms, atom_features)``.
            m_rbf: Per-edge features (RBF in the embedding block, ``m_ca`` in
                interaction blocks), shape ``(nEdges, edge_features)``.
            idnb_a: Source atom index for each edge, shape ``(nEdges,)``.
                The historical name is preserved for API compatibility.
            idnb_c: Target atom index for each edge, shape ``(nEdges,)``.
                The historical name is preserved for API compatibility.

        Returns:
            Edge embeddings of shape ``(nEdges, out_features)``.
        """
        h_src = h[idnb_a]  # (nEdges, atom_features)
        h_dst = h[idnb_c]  # (nEdges, atom_features)
        m_ca = torch.cat([h_src, h_dst, m_rbf], dim=-1)  # (nEdges, 2*atom_features + edge_features)
        m_ca = self.dense(m_ca)  # (nEdges, out_features)
        return m_ca
