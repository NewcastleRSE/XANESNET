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

"""MACE foundation-model descriptor for XANESNET."""

import numpy as np
from ase import Atoms
from mace.calculators import mace_mp

from .base import Descriptor
from .registry import DescriptorRegistry


@DescriptorRegistry.register("mace")
class MACE(Descriptor):
    """MACE foundation-model descriptor.

    Uses the MACE model to compute per-atom equivariant
    features as the structural descriptor.

    Args:
        descriptor_type: Identifier string for this descriptor type.
        invariants_only: If ``True``, return only rotationally invariant
            features.
        num_layers: Number of MACE message-passing layers to use (``-1`` for
            all).
        device: Compute device for the MACE model.
    """

    def __init__(
        self,
        descriptor_type: str,
        invariants_only: bool,
        num_layers: int,
        device: str,
    ) -> None:
        """Initialize ``MACE``."""
        super().__init__(descriptor_type)

        self.invariants_only = invariants_only
        self.num_layers = num_layers
        self.device = device
        self.mace = mace_mp(device=device, default_dtype="float32")

    def transform(
        self,
        system: Atoms,
        site_index: int | list[int] | None = 0,
    ) -> np.ndarray:
        """Compute MACE per-atom descriptors for one or more sites.

        Args:
            system: The atomic system.
            site_index: Site index, list of site indices, or ``None`` for all sites.
                Defaults to ``0`` (the target site).

        Returns:
            Descriptor array ``(S, D)`` where ``S`` is the number of selected sites
            and ``D`` is the MACE feature dimension.
        """
        descriptors = np.asarray(
            self.mace.get_descriptors(
                system,
                invariants_only=self.invariants_only,
                num_layers=self.num_layers,
            )
        )
        if isinstance(site_index, int):
            site_index = [site_index]
        if site_index is not None:
            return descriptors[site_index, :]
        return descriptors
