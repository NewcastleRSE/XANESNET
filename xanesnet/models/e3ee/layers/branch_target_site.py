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

"""Invariant target-site branch for E3EE."""

import torch
import torch.nn as nn

from .basic import MLP


class EnergyConditionedTargetSiteBranch(nn.Module):
    """Energy-dependent branch based on invariant target-site features.

    Args:
        atom_dim: Dimension of the invariant target-site feature vector.
        e_dim: Dimension of the energy RBF embedding.
        hidden_dim: Hidden dimension of the MLP.
        out_dim: Output (latent) dimension.
    """

    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        """Initialize ``EnergyConditionedTargetSiteBranch``."""
        super().__init__()
        self.mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, target_features: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """Compute a latent representation from target-site features and energy.

        Args:
            target_features: Target-site invariant features, shape ``(B, H)``.
            e_feat: Energy RBF features, shape ``(nE, dE)``.

        Returns:
            Latent tensor of shape ``(B, nE, latent_dim)``.
        """
        bsz, h_dim = target_features.shape
        n_energies, e_dim = e_feat.shape

        expanded_target_features = target_features.unsqueeze(1).expand(bsz, n_energies, h_dim)
        expanded_energy_features = e_feat.unsqueeze(0).expand(bsz, n_energies, e_dim)
        return self.mlp(torch.cat([expanded_target_features, expanded_energy_features], dim=-1))
