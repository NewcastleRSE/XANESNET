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
#   ...

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
