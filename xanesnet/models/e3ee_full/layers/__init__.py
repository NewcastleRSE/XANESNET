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

"""Public API for E3EEFull layer modules."""

from .atom_encoder import EquivariantAtomEncoder
from .basic import (
    MLP,
    CosineCutoff,
    EnergyRBFEmbedding,
    GaussianRBF,
    IrrepNorm,
    RadialMLP,
)
from .branch_all_atom import AllAtomEnergyBranch
from .branch_attention import AllAtomAtomAttention
from .branch_convolution import (
    AllAtomAtomConvolution,
    AllAtomEquivariantAtomConvolution,
)
from .branch_eq_attention import AllAtomEquivariantAtomAttention
from .branch_equivariant import AllAtomEquivariantHead, EnergyIrrepModulation
from .branch_fusion import GatedBranchFusion
from .branch_path import AllAtomPathAggregator, PairElementEnergyScattering
from .interactions import EquivariantInteractionBlock

__all__ = [
    "AllAtomAtomAttention",
    "AllAtomAtomConvolution",
    "AllAtomEnergyBranch",
    "AllAtomEquivariantAtomAttention",
    "AllAtomEquivariantAtomConvolution",
    "AllAtomEquivariantHead",
    "AllAtomPathAggregator",
    "CosineCutoff",
    "EnergyIrrepModulation",
    "EnergyRBFEmbedding",
    "EquivariantAtomEncoder",
    "EquivariantInteractionBlock",
    "GaussianRBF",
    "GatedBranchFusion",
    "IrrepNorm",
    "MLP",
    "PairElementEnergyScattering",
    "RadialMLP",
]
