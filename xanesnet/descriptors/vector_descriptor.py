"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

from abc import ABC

from .base import Descriptor

###############################################################################
################################## CLASSES ####################################
###############################################################################


class VectorDescriptor(Descriptor, ABC):
    """
    An abstract base class for transforming a molecular system into a
    fingerprint feature vector, or 'descriptor', that encodes the local
    environment around an absorption site as a vector.
    """

    def __init__(
        self,
        descriptor_type: str,
        r_min: float,
        r_max: float,
        use_charge: bool,
        use_spin: bool,
    ) -> None:
        """
        Args:
            r_min (float): The minimum radial cutoff distance (in A) around
                the absorption site.
            r_max (float): The maximum radial cutoff distance (in A) around
                the absorption site.
            use_charge (bool): If True, includes an additional element in the
                vector descriptor for the charge state of the complex.
                Defaults to False.
            use_spin (bool): If True, includes an additional element in the
                vector descriptor for the spin state of the complex.
                Defaults to False.
        """
        super().__init__(descriptor_type)

        if isinstance(r_min, (int, float)) and r_min >= 0.0:
            self.r_min = float(r_min)
        else:
            raise ValueError(f"expected r_min: int/float >== 0.0; got {r_min}")

        if isinstance(r_max, (int, float)) and r_max > r_min:
            self.r_max = float(r_max)
        else:
            raise ValueError(f"expected r_max: int/float > r_min; got {r_max}")

        if isinstance(use_charge, bool):
            self.use_charge = use_charge
        else:
            raise ValueError(f"expected use_charge: bool; got {use_charge}")

        if isinstance(use_spin, bool):
            self.use_spin = use_spin
        else:
            raise ValueError(f"expected use_spin: bool; got {use_spin}")
