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

"""Dataset package public API."""

from .base import Dataset, TorchDataset, TorchGeometricDataset
from .registry import DatasetRegistry
from .torch import (
    DescriptorData,
    DescriptorDataset,
    DescriptorDatasetMp,
    DescriptorMultiheadData,
    DescriptorMultiheadDataset,
    DescriptorMultiheadDatasetMp,
    EnvEmbedData,
    EnvEmbedDataset,
    EnvEmbedDatasetMp,
)
from .torchgeometric import (
    E3EEBatch,
    E3EEDataset,
    E3EEDatasetMp,
    E3EEFullBatch,
    E3EEFullDataset,
    E3EEFullDatasetMp,
    GemNetBatch,
    GemNetData,
    GemNetDataset,
    GemNetDatasetMp,
    GeometryGraphBatch,
    GeometryGraphData,
    GeometryGraphDataset,
    GeometryGraphDatasetMp,
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
    "GeometryGraphBatch",
    "GeometryGraphData",
    "GeometryGraphDataset",
    "DescriptorDatasetMp",
    "DescriptorMultiheadData",
    "DescriptorMultiheadDataset",
    "DescriptorMultiheadDatasetMp",
    "EnvEmbedDatasetMp",
    "E3EEDatasetMp",
    "E3EEFullDatasetMp",
    "GemNetDatasetMp",
    "GeometryGraphDatasetMp",
]
