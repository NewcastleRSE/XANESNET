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

import os
from typing import Protocol

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ..base import TorchGeometricDataset
from ..registry import DatasetRegistry


# for typing
class GeometricBatch(Protocol):
    x: torch.Tensor
    pos: torch.Tensor
    energies: torch.Tensor
    intensities: torch.Tensor
    sample_id: torch.Tensor
    atomic_symbols: torch.Tensor
    batch: torch.Tensor


###############################################################################
#################################### CLASS ####################################
###############################################################################


@DatasetRegistry.register("geometric")
class GeometricDataset(TorchGeometricDataset):
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

    def prepare(self) -> bool:
        already_processed = super().prepare()

        if already_processed:
            return True

        for idx, pmg_obj in tqdm(enumerate(self.datasource), total=len(self.datasource), desc="Processing data"):
            sample_id = pmg_obj.properties["file_name"]
            atomic_symbols = pmg_obj.labels

            atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
            cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)

            # XANES (first atom)
            # TODO if we want to do multi-absorber training in the future, we would need to store
            # TODO energies and intensities for all atoms and index them in the model forward pass.
            energies, intensities = (
                pmg_obj.site_properties["XANES"][0]["energies"],
                pmg_obj.site_properties["XANES"][0]["intensities"],
            )
            energies = torch.tensor(energies, dtype=torch.float32)
            intensities = torch.tensor(intensities, dtype=torch.float32)

            struct = Data(
                x=atomic_numbers,
                pos=cart_coords,
                energies=energies,
                intensities=intensities,
                sample_id=sample_id,  # TODO maybe rename to file_name for consistency
                atomic_symbols=atomic_symbols,
            )

            save_path = os.path.join(self.processed_dir, f"{idx}.pth")
            self._save_data(struct, save_path)

        return True

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        fields_to_stack = ["energies", "intensities"]
        batched = Batch.from_data_list(batch, exclude_keys=fields_to_stack)
        for field in fields_to_stack:
            setattr(batched, field, torch.stack([getattr(d, field) for d in batch]))
        return batched

    @staticmethod
    def _save_data(data: Data, path: str) -> None:
        tensor_dict = data.to_dict()
        torch.save(tensor_dict, path)

    def _load_item(self, path: str) -> Data:
        tensor_dict = torch.load(path, weights_only=True)
        return Data(**tensor_dict)

    @property
    def signature(self) -> Config:
        """
        Return dataset signature as a dictionary.
        """
        signature = super().signature
        signature.update_with_dict({})
        return signature
