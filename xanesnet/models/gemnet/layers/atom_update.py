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
from torch_scatter import scatter

from ..utils.initializer import he_orthogonal_init
from .base import Dense, ResidualLayer
from .scaling import ScalingFactor


class AtomUpdateBlock(torch.nn.Module):
    """
    Aggregate the message embeddings of the atoms
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
        super().__init__()

        self.name = name
        self.emb_size_edge = emb_size_edge

        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_sum = ScalingFactor(scale_file=scale_file, name=name + "_sum")

        self.layers = self.get_mlp(emb_size_atom, nHidden, activation)

    def get_mlp(self, units: int, nHidden: int, activation: str) -> torch.nn.ModuleList:
        dense1 = Dense(self.emb_size_edge, units, activation=activation, bias=False)
        res = [ResidualLayer(units, nLayers=2, activation=activation) for _ in range(nHidden)]
        mlp = [dense1] + res
        return torch.nn.ModuleList(mlp)

    def reset_parameters(self) -> None:
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
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size_atom) Atom embedding.
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
    """
    Combines the atom update block and subsequent final dense layer.
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
        # do not add bias to final layer to enforce that prediction for an atom
        # without any edge embeddings is zero
        self.out_energy = Dense(emb_size_atom, num_targets, bias=False, activation=None)

    def reset_parameters(self) -> None:
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
        """
        Returns
        -------
            E: torch.Tensor, shape=(nAtoms, num_targets)
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
