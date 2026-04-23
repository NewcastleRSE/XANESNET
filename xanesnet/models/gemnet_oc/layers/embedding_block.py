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

from .base_layers import Dense


class AtomEmbedding(torch.nn.Module):
    """Initial atom embeddings based on the atomic number.

    ``num_elements`` sets the size of the embedding table. The default of 94
    is enough for all naturally occurring elements plus hydrogen's 0-index
    offset (`Z - 1` is used below).
    """

    def __init__(self, emb_size: int, num_elements: int = 94) -> None:
        super().__init__()
        self.emb_size = emb_size

        self.embeddings = torch.nn.Embedding(num_elements, emb_size)
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.embeddings(Z - 1)


class EdgeEmbedding(torch.nn.Module):
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
        h_s = h[edge_index[0]]
        h_t = h[edge_index[1]]
        m_st = torch.cat([h_s, h_t, m], dim=-1)
        return self.dense(m_st)
