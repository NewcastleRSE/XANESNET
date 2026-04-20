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

import logging
import os
from typing import Protocol

import numpy as np
import torch
from pymatgen.core import Molecule, Structure
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.nn import radius_graph
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ..base import TorchGeometricDataset
from ..registry import DatasetRegistry

SPECTRUM_KEYS = ["XANES", "XANES_K"]  # TODO maybe put this somewhere more central?


# for typing
class RadiusGraphBatch(Protocol):
    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    batch: torch.Tensor
    energies: torch.Tensor
    intensities: torch.Tensor
    absorber_mask: torch.Tensor
    file_name: torch.Tensor


###############################################################################
#################################### CLASS ####################################
###############################################################################


@DatasetRegistry.register("radiusgraph")
class RadiusGraphDataset(TorchGeometricDataset):
    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
        force_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
        # params
        cutoff: float,
        max_num_neighbors: int,
    ) -> None:
        super().__init__(dataset_type, datasource, root, preload, force_prepare, split_ratios, split_indexfile)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def prepare(self) -> bool:
        already_processed = super().prepare()

        if already_processed:
            return True

        for idx, pmg_obj in tqdm(enumerate(self.datasource), total=len(self.datasource), desc="Processing data"):
            # Check if XANES spectrum is available for the sample; if not, skip processing
            for key in SPECTRUM_KEYS:
                if key in pmg_obj.site_properties.keys():
                    break
            else:
                logging.warning(
                    f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['file_name']}); skipping."
                )
                continue

            # XANES
            xanes = np.array(pmg_obj.site_properties[key], dtype=object)
            xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()
            xanes = xanes[xanes_idxs]
            absorber_mask = torch.zeros(len(pmg_obj.labels), dtype=torch.bool)
            absorber_mask[xanes_idxs] = True
            intensities = np.array([x["intensities"] for x in xanes], dtype=np.float32)
            energies = np.array([x["energies"] for x in xanes], dtype=np.float32)

            # Atomic numbers and coordinates
            atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
            cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)
            energies = torch.tensor(energies, dtype=torch.float32)
            intensities = torch.tensor(intensities, dtype=torch.float32)

            # Edges
            edge_index, edge_weight = self._build_edges(pmg_obj, self.cutoff, self.max_num_neighbors)

            struct = Data(
                x=atomic_numbers,
                pos=cart_coords,
                edge_index=edge_index,
                edge_weight=edge_weight,
                batch=None,  # will be set in collate_fn
                energies=energies,
                intensities=intensities,
                absorber_mask=absorber_mask,
                file_name=pmg_obj.properties["file_name"],
            )

            save_path = os.path.join(self.processed_dir, f"{idx}.pth")
            self._save_data(struct, save_path)

        return True

    def collate_fn(self, batch: list[BaseData]) -> Batch:
        fields_to_cat = ["energies", "intensities", "absorber_mask"]
        batched = Batch.from_data_list(batch, exclude_keys=fields_to_cat)
        for field in fields_to_cat:
            setattr(batched, field, torch.cat([getattr(d, field) for d in batch], dim=0))
        return batched

    @staticmethod
    def _build_edges(
        pmg_obj: Structure | Molecule,
        cutoff: float,
        max_num_neighbors: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(pmg_obj, Structure):  # Structure
            all_neighbors = pmg_obj.get_all_neighbors(r=cutoff)
            src, dst, dists = [], [], []
            for i, site_neighbors in enumerate(all_neighbors):
                sorted_neighbors = sorted(site_neighbors, key=lambda n: n.nn_distance)
                for neighbor in sorted_neighbors[:max_num_neighbors]:
                    src.append(i)
                    dst.append(neighbor.index)
                    dists.append(neighbor.nn_distance)
            edge_index = torch.tensor([src, dst], dtype=torch.int64)
            edge_weight = torch.tensor(dists, dtype=torch.float32)
            return edge_index, edge_weight
        else:  # Molecule
            pos = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)
            edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=max_num_neighbors)
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            return edge_index, edge_weight

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
