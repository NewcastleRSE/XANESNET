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
from typing import Any, Protocol

import numpy as np
import torch
from pymatgen.core import Molecule, Structure
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.nn import radius_graph
from torch_geometric.typing import SparseTensor
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ..base import TorchGeometricDataset
from ..registry import DatasetRegistry

SPECTRUM_KEYS = ["XANES", "XANES_K"]  # TODO maybe put this somewhere more central?


class RadiusGraphData(Data):
    """
    Custom Data subclass that tells PyG's batching how to handle triplet indices.
    idx_kj and idx_ji are edge-level indices that must be offset by the cumulative
    edge count when batching multiple graphs (similar to how edge_index is offset
    by the cumulative node count).
    """

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in ("idx_kj", "idx_ji"):
            return self.edge_index.size(1)  # type: ignore[union-attr]
        return super().__inc__(key, value, *args, **kwargs)


# for typing
class RadiusGraphBatch(Protocol):
    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    batch: torch.Tensor
    # Triplet fields (only present when compute_angles=True)
    angle: torch.Tensor
    idx_kj: torch.Tensor
    idx_ji: torch.Tensor
    # Targets
    energies: torch.Tensor
    intensities: torch.Tensor
    absorber_mask: torch.Tensor
    file_name: torch.Tensor


###############################################################################
#################################### UTILS ####################################
###############################################################################


def _edges_from_structure(
    structure: Structure,
    cutoff: float,
    max_num_neighbors: int,
    compute_vectors: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Build edge_index, edge_weight (and optionally displacement vectors) for a periodic structure.
    """
    all_neighbors = structure.get_all_neighbors(r=cutoff)
    src, dst, dists = [], [], []
    edge_vectors: list | None = [] if compute_vectors else None

    for i, site_neighbors in enumerate(all_neighbors):
        sorted_neighbors = sorted(site_neighbors, key=lambda n: n.nn_distance)
        for neighbor in sorted_neighbors[:max_num_neighbors]:
            src.append(i)
            dst.append(neighbor.index)
            dists.append(neighbor.nn_distance)
            if edge_vectors is not None:
                edge_vectors.append(neighbor.coords - structure.cart_coords[i])

    edge_index = torch.tensor([src, dst], dtype=torch.int64)
    edge_weight = torch.tensor(dists, dtype=torch.float32)
    edge_vec = torch.tensor(np.array(edge_vectors), dtype=torch.float32) if compute_vectors else None
    return edge_index, edge_weight, edge_vec


def _edges_from_molecule(
    molecule: Molecule,
    cutoff: float,
    max_num_neighbors: int,
    compute_vectors: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Build edge_index, edge_weight (and optionally displacement vectors) for a molecule.
    """
    pos = torch.tensor(molecule.cart_coords, dtype=torch.float32)
    edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=max_num_neighbors)
    row, col = edge_index
    edge_weight = (pos[row] - pos[col]).norm(dim=-1)
    edge_vec = pos[col] - pos[row] if compute_vectors else None  # displacement from source to target
    return edge_index, edge_weight, edge_vec


def _compute_triplets_and_angles(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    num_nodes: int,
    is_periodic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute triplets (k->j->i) and the angle at node j for each triplet.

    Returns (angle, idx_kj, idx_ji).
    """
    row, col = edge_index

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col,
        col=row,
        value=value,
        sparse_sizes=(num_nodes, num_nodes),
    )
    adj_t_row = adj_t.index_select(0, row)  # type: ignore[attr-defined]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets
    idx_i = col.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    # Edge indices (k->j, j->i) for triplets
    idx_kj_raw = adj_t_row.storage.value()
    idx_ji_raw = adj_t_row.storage.row()

    # Remove degenerate triplets.
    if is_periodic:
        # For periodic structures, idx_i == idx_k does NOT imply a degenerate
        # bounce-back: k and i may be different periodic images of the same atom.
        # Only remove triplets where the same edge serves as both legs.
        mask = idx_kj_raw != idx_ji_raw
    else:
        # For molecules, each atom pair has exactly one edge per direction,
        # so idx_i == idx_k correctly identifies bounce-back triplets.
        mask = idx_i != idx_k

    idx_kj = idx_kj_raw[mask]
    idx_ji = idx_ji_raw[mask]

    # Compute the angle at node j (the intermediate node in the triplet k->j->i).
    # vec_ji: j→i displacement, vec_jk: j→k displacement (negated k→j edge).
    vec_ji = edge_vec[idx_ji]
    vec_jk = -edge_vec[idx_kj]

    a = (vec_ji * vec_jk).sum(dim=-1)
    b = torch.cross(vec_ji, vec_jk, dim=1).norm(dim=-1)
    angle = torch.atan2(b, a)

    return angle, idx_kj, idx_ji


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
        skip_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
        # params
        cutoff: float,
        max_num_neighbors: int,
        compute_angles: bool,
    ) -> None:
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.compute_angles = compute_angles

    def prepare(self) -> bool:
        skip_processing = super().prepare()

        if skip_processing:
            return True

        counter = 0  # Counter for naming processed files
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

            edge_index, edge_weight, angle, idx_kj, idx_ji = self._build_edges(
                pmg_obj,
                self.cutoff,
                self.max_num_neighbors,
                self.compute_angles,
            )

            struct = RadiusGraphData(
                x=atomic_numbers,
                pos=cart_coords,
                edge_index=edge_index,
                edge_weight=edge_weight,
                batch=None,  # will be set in collate_fn
                angle=angle,
                idx_kj=idx_kj,
                idx_ji=idx_ji,
                energies=energies,
                intensities=intensities,
                absorber_mask=absorber_mask,
                file_name=pmg_obj.properties["file_name"],
            )

            save_path = os.path.join(self.processed_dir, f"{counter}.pth")
            self._save_data(struct, save_path)
            counter += 1

        self._length = counter
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
        compute_angles: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Build edges (and optionally triplet angles) for the molecular/crystal graph.

        When compute_angles is False, returns:
            (edge_index, edge_weight, None, None, None)
        When compute_angles is True, returns:
            (edge_index, edge_weight, angle, idx_kj, idx_ji)
        where angle, idx_kj, idx_ji correspond to triplets (k->j->i) as used by e.g. DimeNet.
        """
        if isinstance(pmg_obj, Structure):
            edge_index, edge_weight, edge_vec = _edges_from_structure(
                pmg_obj, cutoff, max_num_neighbors, compute_vectors=compute_angles
            )
        else:
            edge_index, edge_weight, edge_vec = _edges_from_molecule(
                pmg_obj, cutoff, max_num_neighbors, compute_vectors=compute_angles
            )

        if not compute_angles:
            return edge_index, edge_weight, None, None, None

        assert edge_vec is not None
        is_periodic = isinstance(pmg_obj, Structure)
        angle, idx_kj, idx_ji = _compute_triplets_and_angles(
            edge_index, edge_vec, num_nodes=len(pmg_obj), is_periodic=is_periodic
        )
        return edge_index, edge_weight, angle, idx_kj, idx_ji

    @staticmethod
    def _save_data(data: Data, path: str) -> None:
        tensor_dict = data.to_dict()
        torch.save(tensor_dict, path)

    def _load_item(self, path: str) -> RadiusGraphData:
        tensor_dict = torch.load(path, weights_only=True)
        return RadiusGraphData(**tensor_dict)

    @property
    def signature(self) -> Config:
        """
        Return dataset signature as a dictionary.
        """
        signature = super().signature
        signature.update_with_dict({})
        return signature
