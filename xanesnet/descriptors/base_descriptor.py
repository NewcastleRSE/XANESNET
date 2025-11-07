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

from typing import Dict

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np

from ase import Atoms
from abc import ABC, abstractmethod

###############################################################################
################################## CLASSES ####################################
###############################################################################


class BaseDescriptor(ABC):
    """Abstract base class for XANESNET descriptors."""

    def __init__(self):
        self.config = {}

    @abstractmethod
    def transform(self, system: Atoms) -> np.ndarray:
        """
        Args:
            system (Atoms): A molecular system.

        Returns:
            np.ndarray: A fingerprint feature vector for the molecular system.
        """

        pass

    @abstractmethod
    def get_nfeatures(self) -> int:
        """
        Return:
            int: Number of features for this descriptor.
        """

        pass

    @abstractmethod
    def get_type(self) -> str:
        """
        Return:
            str: descriptor type
        """

        pass

    def register_config(self, args, **kwargs):
        """
        Assign arguments from the child class's constructor to self.config.

        Args:
            args: The dictionary of arguments from the child class's constructor
            **kwargs: additional arguments to store
        """
        config = kwargs.copy()

        # Extract parameters from the local_vars, excluding 'self' and '__class__'
        args_dict = {
            key: val for key, val in args.items() if key not in ["self", "__class__"]
        }

        config.update(args_dict)

        self.config = config
