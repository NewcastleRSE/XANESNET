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

import torch
from torch.utils.data._utils.collate import default_collate

from xanesnet.datasources import DataSource
from xanesnet.utils.mode import Mode

from .base import Dataset

###############################################################################
#################################### CLASS ####################################
###############################################################################


class TorchDataset(Dataset, torch.utils.data.Dataset):
    """
    A dataset class that combines BaseDataset and PyTorch's Dataset.
    This class can be used to create datasets compatible with PyTorch's DataLoader.
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

    def get_dataloader(self) -> type[torch.utils.data.DataLoader]:
        """
        Returns the dataloader class that should be used.
        """
        return torch.utils.data.DataLoader

    def collate_fn(self, batch: list[Any]) -> Any:
        """
        Uses the default collate function from pytorch.
        """
        return default_collate(batch)
