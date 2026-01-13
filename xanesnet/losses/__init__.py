"""
XANESNET

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

from .base import Loss
from .bcewithlogits import BCEWithLogitsLoss
from .emd import EMDLoss
from .l1 import L1Loss
from .mse import MSELoss
from .mwssim import MultiWindowSSIMLoss
from .registry import LossRegistry
from .specplus import SpectralLossPlus
from .wcc import WCCLoss

__all__ = [
    "Loss",
    "LossRegistry",
    "BCEWithLogitsLoss",
    "EMDLoss",
    "L1Loss",
    "MSELoss",
    "MultiWindowSSIMLoss",
    "SpectralLossPlus",
    "WCCLoss",
]
