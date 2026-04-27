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

"""Atom and output update blocks aggregating edge messages into atom embeddings."""

import math

import torch
from torch_scatter import scatter

from .base_layers import Dense, ResidualLayer
from .scaling import ScaleFactor


class AtomUpdateBlock(torch.nn.Module):
    """Aggregate edge messages into atom embeddings and apply a residual MLP.

    Args:
        emb_size_atom: Atom embedding dimension (output).
        emb_size_edge: Edge embedding dimension.
        emb_size_rbf: Radial basis function dimension.
        nHidden: Number of :class:`ResidualLayer` blocks in the per-atom MLP.
        activation: Activation function name.
    """

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

    def get_mlp(
        self, units_in: int, units: int, nHidden: int, activation: str | None
    ) -> torch.nn.ModuleList:
        """Build the per-atom MLP as a :class:`~torch.nn.ModuleList`.

        Prepends a projection layer when ``units_in != units``.

        Args:
            units_in: Input embedding dimension.
            units: Output (and hidden) embedding dimension.
            nHidden: Number of :class:`ResidualLayer` blocks.
            activation: Activation function name.

        Returns:
            A :class:`~torch.nn.ModuleList` of dense and residual layers.
        """
        if units_in != units:
            dense1 = Dense(units_in, units, activation=activation, bias=False)
            mlp: list[Dense | ResidualLayer] = [dense1]
        else:
            mlp = []
        res = [ResidualLayer(units, nLayers=2, activation=activation) for _ in range(nHidden)]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(
        self, h: torch.Tensor, m: torch.Tensor, basis_rad: torch.Tensor, idx_atom: torch.Tensor
    ) -> torch.Tensor:
        """Update atom embeddings by aggregating scaled edge messages.

        Args:
            h: Current atom embeddings, shape ``(nAtoms, emb_size_atom)``.
            m: Edge message embeddings, shape ``(nEdges, emb_size_edge)``.
            basis_rad: Radial basis values for each edge,
                shape ``(nEdges, emb_size_rbf)``.
            idx_atom: Target atom index for each edge, shape ``(nEdges,)``.

        Returns:
            Updated atom embeddings of shape ``(nAtoms, emb_size_atom)``.
        """
        nAtoms = h.shape[0]

        bases_emb = self.dense_rbf(basis_rad)  # (nEdges, emb_size_edge)
        x = m * bases_emb

        x2 = scatter(x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum")
        x = self.scale_sum(x2, ref=m)

        for layer in self.layers:
            x = layer(x)
        return x


class OutputBlock(AtomUpdateBlock):
    """XANES output block that returns a per-atom embedding (no force branch).

    Extends :class:`AtomUpdateBlock` with an optional second residual MLP
    fused with the atom skip connection.

    Args:
        emb_size_atom: Atom embedding dimension.
        emb_size_edge: Edge embedding dimension.
        emb_size_rbf: Radial basis function dimension.
        nHidden: Number of :class:`ResidualLayer` blocks in the first MLP.
        nHidden_afteratom: Number of :class:`ResidualLayer` blocks in the
            second MLP (applied after the atom skip connection). Set to
            ``0`` to disable.
        activation: Activation function name.
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

    def forward(
        self, h: torch.Tensor, m: torch.Tensor, basis_rad: torch.Tensor, idx_atom: torch.Tensor
    ) -> torch.Tensor:
        """Compute the output atom embedding.

        Args:
            h: Current atom embeddings, shape ``(nAtoms, emb_size_atom)``.
            m: Edge message embeddings, shape ``(nEdges, emb_size_edge)``.
            basis_rad: Radial basis values for each edge,
                shape ``(nEdges, emb_size_rbf)``.
            idx_atom: Target atom index for each edge, shape ``(nEdges,)``.

        Returns:
            Output atom embeddings of shape ``(nAtoms, emb_size_atom)``.
        """
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
