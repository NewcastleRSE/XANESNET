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

from .base import Dataset, TorchDataset, TorchGeometricDataset
from .registry import DatasetRegistry
from .torch import (
    EnvEmbedData,
    EnvEmbedDataset,
    GemNetBatch,
    GemNetData,
    GemNetDataset,
    XanesXData,
    XanesXDataset,
)
from .torchgeometric import (
    E3EEBatch,
    E3EEDataset,
    RadiusGraphBatch,
    RadiusGraphDataset,
    RichGraphDataset,
)

__all__ = [
    "Dataset",
    "TorchDataset",
    "TorchGeometricDataset",
    "XanesXDataset",
    "DatasetRegistry",
    "XanesXData",
    "GemNetDataset",
    "GemNetData",
    "GemNetBatch",
    "E3EEDataset",
    "EnvEmbedData",
    "EnvEmbedDataset",
    "E3EEBatch",
    "RichGraphDataset",
    "RadiusGraphBatch",
    "RadiusGraphDataset",
]
