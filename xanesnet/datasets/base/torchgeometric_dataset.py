# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either Version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

"""PyTorch Geometric dataset adapter."""

import torch_geometric.data as tgdata
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from xanesnet.datasources import DataSource

from .base import Dataset


class TorchGeometricDataset(Dataset, tgdata.Dataset):
    """Dataset base class for PyTorch Geometric dataloaders.

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw data source used during preparation.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
    """

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
        skip_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
    ) -> None:
        """Initialize a PyTorch Geometric dataset."""
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        """Collate graph samples into a PyG batch.

        Args:
            batch: Graph samples loaded by ``__getitem__``.

        Returns:
            PyG batch created by ``Batch.from_data_list``.
        """
        return Batch.from_data_list(batch)
