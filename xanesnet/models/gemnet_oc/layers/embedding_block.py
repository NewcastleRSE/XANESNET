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

"""Atom and edge embedding modules for GemNet-OC."""

import numpy as np
import torch

from .base_layers import Dense


class AtomEmbedding(torch.nn.Module):
    """Initial atom embeddings looked up by atomic number.

    Args:
        emb_size: Atom embedding dimension.
        num_elements: Embedding table size. The default of 94 covers atomic
            numbers 1-94 (Hydrogen to Plutonium). Uses ``Z - 1`` indexing.
    """

    def __init__(self, emb_size: int, num_elements: int = 94) -> None:
        super().__init__()
        self.emb_size = emb_size

        self.embeddings = torch.nn.Embedding(num_elements, emb_size)
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Look up atom embeddings by atomic number.

        Args:
            Z: Atomic numbers, shape ``(nAtoms,)``.

        Returns:
            Atom embeddings of shape ``(nAtoms, emb_size)``.
        """
        return self.embeddings(Z - 1)


class EdgeEmbedding(torch.nn.Module):
    """Edge embeddings from concatenated atom embeddings and edge features.

    Args:
        atom_features: Atom embedding dimension.
        edge_features: Per-edge feature dimension (e.g. number of radial basis
            functions).
        out_features: Output edge embedding dimension.
        activation: Activation function name.
    """

    def __init__(
        self,
        atom_features: int,
        edge_features: int,
        out_features: int,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(self, h: torch.Tensor, m: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute edge embeddings from atom and edge features.

        Args:
            h: Atom embeddings, shape ``(nAtoms, atom_features)``.
            m: Per-edge features (e.g. RBF), shape ``(nEdges, edge_features)``.
            edge_index: Edge connectivity, shape ``(2, nEdges)``, where row 0
                is the source atom and row 1 is the target atom.

        Returns:
            Edge embeddings of shape ``(nEdges, out_features)``.
        """
        h_s = h[edge_index[0]]
        h_t = h[edge_index[1]]
        m_st = torch.cat([h_s, h_t, m], dim=-1)
        return self.dense(m_st)
