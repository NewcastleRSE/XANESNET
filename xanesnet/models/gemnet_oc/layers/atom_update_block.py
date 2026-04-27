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

import math

import torch
from torch_scatter import scatter

from .base_layers import Dense, ResidualLayer
from .scaling import ScaleFactor


class AtomUpdateBlock(torch.nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation=None,
    ) -> None:
        super().__init__()

        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_sum = ScaleFactor()

        self.layers = self.get_mlp(emb_size_edge, emb_size_atom, nHidden, activation)

    def get_mlp(self, units_in: int, units: int, nHidden: int, activation):
        if units_in != units:
            dense1 = Dense(units_in, units, activation=activation, bias=False)
            mlp: list[Dense | ResidualLayer] = [dense1]
        else:
            mlp = []
        res = [ResidualLayer(units, nLayers=2, activation=activation) for _ in range(nHidden)]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(self, h: torch.Tensor, m: torch.Tensor, basis_rad: torch.Tensor, idx_atom: torch.Tensor):
        nAtoms = h.shape[0]

        bases_emb = self.dense_rbf(basis_rad)  # (nEdges, emb_size_edge)
        x = m * bases_emb

        x2 = scatter(x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum")
        x = self.scale_sum(x2, ref=m)

        for layer in self.layers:
            x = layer(x)
        return x


class OutputBlock(AtomUpdateBlock):
    """
    XANES output block — returns per-atom embedding only (no force branch).
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        nHidden_afteratom: int,
        activation: str | None = None,
    ) -> None:
        super().__init__(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
        )

        self.seq_energy_pre = self.layers
        if nHidden_afteratom >= 1:
            self.seq_energy2 = self.get_mlp(emb_size_atom, emb_size_atom, nHidden_afteratom, activation)
            self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        else:
            self.seq_energy2 = None

    def forward(self, h: torch.Tensor, m: torch.Tensor, basis_rad: torch.Tensor, idx_atom: torch.Tensor):
        nAtoms = h.shape[0]

        basis_emb_E = self.dense_rbf(basis_rad)
        x = m * basis_emb_E

        x_E = scatter(x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum")
        x_E = self.scale_sum(x_E, ref=m)

        for layer in self.seq_energy_pre:
            x_E = layer(x_E)

        if self.seq_energy2 is not None:
            x_E = x_E + h
            x_E = x_E * self.inv_sqrt_2
            for layer in self.seq_energy2:
                x_E = layer(x_E)

        return x_E
