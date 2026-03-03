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

from .base import Dense


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size: int) -> None:
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Pu (94). Use 93 dimensions because of 0-based indexing
        # TODO Check if this is sufficient for all datasets!
        self.embeddings = torch.nn.Embedding(93, emb_size)

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(z - 1)  # -1 because z.min()=1 (==Hydrogen)
        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        atom_features: int
            Embedding size of the atom embeddings.
        edge_features: int
            Embedding size of the edge embeddings.
        out_features: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features: int,
        edge_features: int,
        out_features: int,
        activation: str,
    ) -> None:
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def reset_parameters(self) -> None:
        self.dense.reset_parameters()

    def forward(
        self,
        h: torch.Tensor,
        m_rbf: torch.Tensor,
        idnb_a: torch.Tensor,
        idnb_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        # m_rbf: shape (nEdges, nFeatures)
        # in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_ca

        h_a = h[idnb_a]  # shape=(nEdges, emb_size)
        h_c = h[idnb_c]  # shape=(nEdges, emb_size)

        m_ca = torch.cat([h_a, h_c, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_ca = self.dense(m_ca)  # (nEdges, emb_size)
        return m_ca
