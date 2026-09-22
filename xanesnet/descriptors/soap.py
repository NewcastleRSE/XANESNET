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

"""Smooth overlap of atomic positions (SOAP) descriptor for XANESNET."""

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP as DscribeSOAP

from .base import Descriptor
from .registry import DescriptorRegistry


@DescriptorRegistry.register("soap")
class SOAP(Descriptor):
    """Smooth overlap of atomic positions (SOAP) descriptor.

    Encodes the local geometry around a site using a SOAP power spectrum
    computed via the dscribe library.

    References:
        Bartók, A. P., Kondor, R., & Csányi, G. (2013).
        On representing chemical environments.
        Physical Review B, 87(18). doi:10.1103/physrevb.87.184115

    Args:
        descriptor_type: Identifier string for this descriptor type.
        r_cut: Local environment cutoff radius. **A**.
        n_max: Number of radial basis functions.
        l_max: Maximum angular momentum quantum number.
        sigma: Gaussian broadening width. **A**.
        species: Atomic numbers to include as distinct species.
            ``None`` to use all elements H (1) through Lr (103).
        average: Averaging mode across atomic centers (``"off"``,
                    ``"inner"``, or ``"outer"``).
        compression_mode: SOAP compression mode (``"off"``, ``"mu2"``,
            ``"crossover"``, or ``"mu1nu1"``).
        compression_species_weighting: Species-weighting dictionary
            for ``compression_mode="species_weighting"``; ``None`` for
            dscribe defaults.
        use_charge: Append charge state scalar to the descriptor.
        use_spin: Append spin state scalar to the descriptor.
    """

    def __init__(
        self,
        descriptor_type: str,
        r_cut: float,
        n_max: int,
        l_max: int,
        sigma: float,
        species: list[int] | None,
        average: str,
        compression_mode: str,
        compression_species_weighting: dict | None,
        use_charge: bool,
        use_spin: bool,
    ) -> None:
        super().__init__(descriptor_type)

        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.sigma = sigma
        self.average = average
        self.compression_mode = compression_mode
        self.compression_species_weighting = compression_species_weighting
        self.use_charge = use_charge
        self.use_spin = use_spin

        if species is None:
            species = list(range(1, 104))

        self._species = tuple(species)
        self._soap_nonperiodic: DscribeSOAP | None = None
        self._soap_periodic: DscribeSOAP | None = None

    def _get_soap(self, periodic: bool) -> DscribeSOAP:
        if periodic:
            if self._soap_periodic is None:
                self._soap_periodic = self._build_soap(periodic=True)
            return self._soap_periodic
        else:
            if self._soap_nonperiodic is None:
                self._soap_nonperiodic = self._build_soap(periodic=False)
            return self._soap_nonperiodic

    def _build_soap(self, periodic: bool) -> DscribeSOAP:
        compression: dict = {"mode": self.compression_mode}
        if self.compression_species_weighting is not None:
            compression["species_weighting"] = self.compression_species_weighting

        return DscribeSOAP(
            species=list(self._species),
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
            periodic=periodic,
            average=self.average,
            compression=compression,
        )

    def transform(
        self,
        system: Atoms,
        site_index: int | list[int] | None = 0,
    ) -> np.ndarray:
        """Compute SOAP descriptors for one or more sites.

        Args:
            system: The atomic system.
            site_index: Site index, list of site indices, or ``None`` for all
                sites.  Defaults to ``0`` (the target site).

        Returns:
            Descriptor array ``(S, F)`` where ``S`` is the number of selected
            sites and ``F`` is the SOAP feature dimension.
        """
        if isinstance(site_index, int):
            site_index = [site_index]

        periodic = bool(system.pbc.any())
        soap = self._get_soap(periodic)
        descriptors = np.atleast_2d(np.asarray(soap.create(system)))
        if site_index is not None:
            descriptors = descriptors[site_index, :]

        if self.use_spin:
            spin = np.full((descriptors.shape[0], 1), system.info.get("S", 0))
            descriptors = np.hstack([descriptors, spin])

        if self.use_charge:
            charge = np.full((descriptors.shape[0], 1), system.info.get("q", 0))
            descriptors = np.hstack([descriptors, charge])

        return descriptors
