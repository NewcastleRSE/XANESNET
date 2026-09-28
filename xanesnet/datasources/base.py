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

"""Abstract base class defining the uniform data-loading interface for all XANESNET data sources.

Every ``DataSource`` returns a :class:`pymatgen.core.Molecule` or
:class:`pymatgen.core.Structure` with the following normalized shape:

* ``properties["sample_id"]`` -- the unique sample identifier, always present.
* When the datasource carries spectral data, the spectrum is attached via
  ``add_site_property("spectrum", ...)`` as a **per-site** list.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from pymatgen.core import Molecule, Structure


class DataSource(ABC):
    """Abstract base class for all XANESNET data sources.

    Args:
        datasource_type: Identifier string for the concrete datasource type.
    """

    def __init__(
        self,
        datasource_type: str,
    ) -> None:
        """Initialize ``DataSource``."""
        self.datasource_type = datasource_type

    @abstractmethod
    def __iter__(self) -> Iterator[Molecule | Structure]:
        """Iterate over all entries in the datasource in index order.

        Returns:
            Iterator yielding :class:`~pymatgen.core.Molecule` or
            :class:`~pymatgen.core.Structure` entries.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of entries available in the datasource.

        Returns:
            Number of available datasource entries.
        """
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Molecule | Structure:
        """Load and return the entry at the given zero-based index.

        Args:
            idx: Zero-based index into the datasource.

        Returns:
            The pymatgen :class:`~pymatgen.core.Molecule` or
            :class:`~pymatgen.core.Structure` at position ``idx``.

        Raises:
            IndexError: If *idx* is out of range.
            ResourceError: If the underlying file cannot be read or parsed.
        """
        ...
