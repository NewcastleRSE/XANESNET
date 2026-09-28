# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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

"""Shared graph-construction utilities used by :class:`GraphBuilder` subclasses.

The functions here operate on already-built edge lists (symmetrisation,
truncation) and on already-built graphs (target-site-centered paths, triplet
angles, and direction-aware higher-order indices). They are model-agnostic
and reused across builders and downstream dataset code.
"""

from .directional_indices import (
    compute_id_swap,
    compute_mixed_triplets,
    compute_quadruplets,
    compute_triplets,
)
from .symmetrize import symmetrize_directed_edges, truncate_per_source
from .target_site_paths import build_target_site_paths
from .triplets import compute_triplets_and_angles

__all__ = [
    "build_target_site_paths",
    "compute_id_swap",
    "compute_mixed_triplets",
    "compute_quadruplets",
    "compute_triplets",
    "compute_triplets_and_angles",
    "symmetrize_directed_edges",
    "truncate_per_source",
]
