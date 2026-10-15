# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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

"""Dense invariant energy-dependent branch applied to every atom in E3EEFull."""

import torch
import torch.nn as nn

from .basic import MLP


class AllAtomEnergyBranch(nn.Module):
    """Energy-dependent branch applied to every atom's invariant features."""

    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        """Initialize ``AllAtomEnergyBranch``."""
        super().__init__()
        self.mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, h_all: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """Compute per-(atom, energy) latent vectors.

        Args:
            h_all: Invariant features for every atom, shape ``(B, N, H)``.
            e_feat: Energy RBF features, shape ``(nE, dE)``.

        Returns:
            Latent tensor of shape ``(B, N, nE, out_dim)``.
        """
        bsz, n_atoms, h_dim = h_all.shape
        n_energies, e_dim = e_feat.shape

        expanded_atoms = h_all.unsqueeze(2).expand(bsz, n_atoms, n_energies, h_dim)
        expanded_energy = e_feat.view(1, 1, n_energies, e_dim).expand(bsz, n_atoms, n_energies, e_dim)
        return self.mlp(torch.cat([expanded_atoms, expanded_energy], dim=-1))
