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

"""Work-in-progress rich graph dataset implementation."""

import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ..base import SavePathFn, TorchGeometricDataset
from ..registry import DatasetRegistry


@DatasetRegistry.register("richgraph")
class RichGraphDataset(TorchGeometricDataset):
    """Work-in-progress graph dataset with radius-graph connectivity.

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw datasource of pymatgen structures or molecules.
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
        """Initialize the rich graph dataset."""
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into one rich graph sample.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps sample sequence numbers to output paths.

        Returns:
            Number of processed graph samples written.
        """
        pmg_obj = self.datasource[idx]
        file_name = pmg_obj.properties["file_name"]
        atomic_symbols = pmg_obj.labels
        atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
        cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)

        # TODO add cutoff_radius config
        edge_index = radius_graph(cart_coords, r=5.0, loop=False)

        row, col = edge_index
        dist = torch.norm(cart_coords[row] - cart_coords[col], dim=1)
        edge_weight = 1 / dist
        edge_weight = edge_weight.view(-1, 1)

        # TODO we need to decide on node, edge, and global features.
        x = torch.tensor([1.0])
        edge_attr = dist.view(-1, 1)
        global_attr = torch.tensor([1.0])

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

        self._save_data(data, save_path_fn(0))
        return 1

    @staticmethod
    def _save_data(data: Data, path: str) -> None:
        """Save one PyG data object as a tensor dictionary.

        Args:
            data: Data object to serialize.
            path: Destination ``.pth`` path.
        """
        tensor_dict = data.to_dict()
        torch.save(tensor_dict, path)

    def _load_item(self, path: str) -> Data:
        """Load one processed rich graph sample.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Reconstructed PyG data object.
        """
        tensor_dict = torch.load(path, weights_only=True)
        return Data(**tensor_dict)

    @property
    def signature(self) -> Config:
        """Dataset configuration signature.

        Returns:
            Configuration values that identify this rich graph dataset.
        """
        signature = super().signature
        signature.update_with_dict({})
        return signature
