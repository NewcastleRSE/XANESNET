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

"""
! ---
! This Dataset is currently a work in progress and not yet fully implemented.
! ---
"""

import os

import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ..base import TorchGeometricDataset
from ..registry import DatasetRegistry

###############################################################################
#################################### CLASS ####################################
###############################################################################


@DatasetRegistry.register("richgraph")
class RichGraphDataset(TorchGeometricDataset):

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
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

    def prepare(self) -> bool:
        skip_processing = super().prepare()

        if skip_processing:
            return True

        counter = 0  # Counter for naming processed files
        for idx, pmg_obj in tqdm(enumerate(self.datasource), total=len(self.datasource), desc="Processing data"):
            file_name = pmg_obj.properties["file_name"]
            atomic_symbols = pmg_obj.labels
            atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
            cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)

            # edge indices
            # TODO add cutoff_radius config
            edge_index = radius_graph(cart_coords, r=5.0, loop=False)

            # edge weights (inverse distance)
            row, col = edge_index
            dist = torch.norm(cart_coords[row] - cart_coords[col], dim=1)
            edge_weight = 1 / dist  # invert to make shorter distances a larger weight
            edge_weight = edge_weight.view(-1, 1)

            # node features
            # TODO we need to decide on node features
            x = torch.tensor([1.0])  # placeholder

            # edge features
            # TODO we need to decide on edge features
            edge_attr = dist.view(-1, 1)

            # global features
            # TODO we need to decide on global features
            global_attr = torch.tensor([1.0])  # placeholder

            # XANES (first atom)
            # TODO if we want to do multi-absorber training in the future, we would need to store
            # TODO energies and intensities for all atoms and index them in the model forward pass.
            energies, intensities = (
                pmg_obj.site_properties["XANES"][0]["energies"],
                pmg_obj.site_properties["XANES"][0]["intensities"],
            )
            energies = torch.tensor(energies, dtype=torch.float32)
            intensities = torch.tensor(intensities, dtype=torch.float32)

            data = Data(
                z=atomic_numbers,
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_attr=edge_attr,
                global_attr=global_attr,
                energies=energies,
                intensities=intensities,
                file_name=file_name,
                atomic_symbols=atomic_symbols,
            )

            save_path = os.path.join(self.processed_dir, f"{counter}.pth")
            self._save_data(data, save_path)
            counter += 1

        self._length = counter
        return True

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
