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

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from xanesnet.datasources import DataSource

from .geometric import GeometricDataset
from .registry import DatasetRegistry

###############################################################################
#################################### CLASS ####################################
###############################################################################


@DatasetRegistry.register("e3ee")
class E3EEDataset(GeometricDataset):
    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
        force_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
    ) -> None:
        super().__init__(dataset_type, datasource, root, preload, force_prepare, split_ratios, split_indexfile)

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        """
        e3ee-specific collate function that does not flatten any tensors.
        """
        fields_to_stack = ["energies", "intensities", "x", "pos"]
        batched = Batch.from_data_list(batch, exclude_keys=fields_to_stack)

        x_list = [sample.x for sample in batch]
        pos_list = [sample.pos for sample in batch]
        mask_list = [torch.ones_like(sample.x, dtype=torch.bool) for sample in batch]
        intensities_list = [sample.intensities for sample in batch]
        energies_list = [sample.energies for sample in batch]

        x = pad_sequence(x_list, batch_first=True, padding_value=0)
        pos = pad_sequence(pos_list, batch_first=True, padding_value=0.0)
        mask = pad_sequence(mask_list, batch_first=True, padding_value=False).to(dtype=torch.bool)
        intensities = torch.stack([inten.to(dtype=torch.float32) for inten in intensities_list], dim=0)
        energies = torch.stack([en.to(dtype=torch.float32) for en in energies_list], dim=0)

        setattr(batched, "x", x)
        setattr(batched, "pos", pos)
        setattr(batched, "mask", mask)
        setattr(batched, "intensities", intensities)
        setattr(batched, "energies", energies)

        return batched
