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

from typing import Any

from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader

from xanesnet.datasources import DataSource
from xanesnet.utils import Mode

from .base import Dataset

###############################################################################
#################################### CLASS ####################################
###############################################################################


class TorchGeometricDataset(
    Dataset,
):
    """
    A dataset class that combines BaseDataset and PyTorch Geometric's Dataset.
    This class can be used to create datasets compatible with PyTorch Geometric's data handling.
    """

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        mode: Mode,
        preload: bool,
        params: dict[str, Any],
    ) -> None:
        super().__init__(dataset_type, datasource, root, mode, preload, params)

    def get_dataloader(self) -> type[DataLoader]:
        """
        Returns the dataloader class that should be used.
        """
        return DataLoader

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        """
        Uses the default collate function from torch_geometric.
        """
        return Batch.from_data_list(batch)
