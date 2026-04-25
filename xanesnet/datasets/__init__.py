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
    DescriptorData,
    DescriptorDataset,
    EnvEmbedData,
    EnvEmbedDataset,
    DescriptorDatasetMp,
    EnvEmbedDatasetMp,
)
from .torchgeometric import (
    E3EEBatch,
    E3EEDataset,
    E3EEFullBatch,
    E3EEFullDataset,
    GemNetBatch,
    GemNetData,
    GemNetDataset,
    GeometryGraphBatch,
    GeometryGraphData,
    GeometryGraphDataset,
    E3EEDatasetMp,
    E3EEFullDatasetMp,
    GemNetDatasetMp,
    GeometryGraphDatasetMp,
    RichGraphDataset,
)

__all__ = [
    "Dataset",
    "TorchDataset",
    "TorchGeometricDataset",
    "DescriptorDataset",
    "DatasetRegistry",
    "DescriptorData",
    "GemNetDataset",
    "GemNetData",
    "GemNetBatch",
    "E3EEDataset",
    "EnvEmbedData",
    "EnvEmbedDataset",
    "E3EEBatch",
    "E3EEFullBatch",
    "E3EEFullDataset",
    "RichGraphDataset",
    "GeometryGraphBatch",
    "GeometryGraphData",
    "GeometryGraphDataset",
    "DescriptorDatasetMp",
    "EnvEmbedDatasetMp",
    "E3EEDatasetMp",
    "E3EEFullDatasetMp",
    "GemNetDatasetMp",
    "GeometryGraphDatasetMp",
]
