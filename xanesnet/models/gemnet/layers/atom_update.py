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

import torch
from torch_scatter import scatter

from ..utils.initializer import he_orthogonal_init
from .base import Dense, ResidualLayer
from .scaling import ScalingFactor


class AtomUpdateBlock(torch.nn.Module):
    """Aggregate edge message embeddings into updated atom embeddings.

    Args:
        emb_size_atom: Atom embedding dimension (output).
        emb_size_edge: Edge embedding dimension.
        emb_size_rbf: Radial basis embedding dimension.
        nHidden: Number of residual layers in the MLP.
        activation: Activation function name.
        scale_file: Path to JSON scale-factor file, or ``None``.
        name: Block name (used to look up scale factors).
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation: str,
        scale_file: str | None,
        name: str,
    ) -> None:
        """Initialize ``AtomUpdateBlock``."""
        super().__init__()

        self.name = name
        self.emb_size_edge = emb_size_edge

        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_sum = ScalingFactor(scale_file=scale_file, name=name + "_sum")

        self.layers = self.get_mlp(emb_size_atom, nHidden, activation)

    def get_mlp(
        self, units: int, nHidden: int, activation: str
    ) -> torch.nn.ModuleList:
        """Build the atom-update MLP: one dense layer followed by residual blocks.

        Args:
            units: Feature dimension for the residual layers.
            nHidden: Number of residual layers.
            activation: Activation function name.

        Returns:
            :class:`~torch.nn.ModuleList` containing the dense layer and
            residual blocks.
        """
        dense1 = Dense(self.emb_size_edge, units, activation=activation, bias=False)
        res = [ResidualLayer(units, nLayers=2, activation=activation) for _ in range(nHidden)]
        mlp = [dense1] + res
        return torch.nn.ModuleList(mlp)

    def reset_parameters(self) -> None:
        """Re-initialize all sub-layer weights."""
        self.dense_rbf.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        rbf: torch.Tensor,
        id_j: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate edge messages and update atom embeddings.

        Args:
            h: Current atom embeddings, shape ``(nAtoms, emb_size_atom)``
                (used only to determine ``nAtoms``).
            m: Edge embeddings, shape ``(nEdges, emb_size_edge)``.
            rbf: Projected radial basis features, shape ``(nEdges, emb_size_rbf)``.
            id_j: Target atom index for each edge, shape ``(nEdges,)``.

        Returns:
            Updated atom embeddings of shape ``(nAtoms, emb_size_atom)``.
        """
        nAtoms = h.shape[0]

        mlp_rbf = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * mlp_rbf

        x2 = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="add")
        x = self.scale_sum(m, x2)  # (nAtoms, emb_size_edge)

        for layer in self.layers:
            x = layer(x)  # (nAtoms, emb_size_atom)
        return x


class OutputBlock(AtomUpdateBlock):
    """Atom update block followed by a final dense projection to target outputs.

    Extends :class:`AtomUpdateBlock` by adding a linear output layer that
    projects atom embeddings to the prediction target dimension.

    Args:
        emb_size_atom: Atom embedding dimension.
        emb_size_edge: Edge embedding dimension.
        emb_size_rbf: Radial basis embedding dimension.
        nHidden: Number of residual layers in the shared MLP.
        num_targets: Output dimension per atom.
        activation: Activation function name.
        output_init: Initialization strategy for the output weight matrix.
            Either ``"HeOrthogonal"`` or ``"zeros"``.
        scale_file: Path to JSON scale-factor file, or ``None``.
        name: Block name (used to look up scale factors).
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        num_targets: int,
        activation: str,
        output_init: str,
        scale_file: str | None,
        name: str,
    ) -> None:

        """Initialize ``OutputBlock``."""
        super().__init__(
            name=name,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
            scale_file=scale_file,
        )

        assert isinstance(output_init, str)
        self.output_init = output_init

        self.seq_energy = self.layers  # inherited from parent class
        # do not add bias to final layer to enforce that prediction for an atom without any edge embeddings is zero
        self.out_energy = Dense(emb_size_atom, num_targets, bias=False, activation=None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialize all weights, applying the configured output initializer."""
        super().reset_parameters()
        if self.output_init.lower() == "heorthogonal":
            he_orthogonal_init(self.out_energy.weight)
        elif self.output_init.lower() == "zeros":
            torch.nn.init.zeros_(self.out_energy.weight)
        else:
            raise ValueError(f"Unknown output_init: {self.output_init}")

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        rbf: torch.Tensor,
        id_j: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate edge messages and return per-atom target predictions.

        Args:
            h: Current atom embeddings, shape ``(nAtoms, emb_size_atom)``
                (used only to determine ``nAtoms``).
            m: Edge embeddings, shape ``(nEdges, emb_size_edge)``.
            rbf: Projected radial basis features, shape ``(nEdges, emb_size_rbf)``.
            id_j: Target atom index for each edge, shape ``(nEdges,)``.

        Returns:
            Per-atom predictions of shape ``(nAtoms, num_targets)``.
        """
        nAtoms = h.shape[0]

        rbf_mlp = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * rbf_mlp

        x_E = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="add")  # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(m, x_E)

        for layer in self.seq_energy:
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        x_E = self.out_energy(x_E)  # (nAtoms, num_targets)

        return x_E
