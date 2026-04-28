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
from typing import Any, Protocol

import numpy as np
import torch
from pymatgen.core import Molecule, Structure
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config
from xanesnet.utils.graph import build_edges, compute_triplets_and_angles

from ..base import SavePathFn, TorchGeometricDataset
from ..registry import DatasetRegistry

SPECTRUM_KEYS = ["XANES", "XANES_K"]  # TODO maybe put this somewhere more central?

###############################################################################
################################ DATA CONTAINER ###############################
###############################################################################


class GeometryGraphData(Data):
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
class GeometryGraphBatch(Protocol):
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
    file_name: list[str]


###############################################################################
#################################### CLASS ####################################
###############################################################################


@DatasetRegistry.register("geometrygraph")
class GeometryGraphDataset(TorchGeometricDataset):
    """
    Geometry-based graph dataset. Supports two edge construction methods:

    - ``graph_method="radius"``: distance-cutoff radius graph.
    - ``graph_method="voronoi"``: Voronoi-tessellation graph (still bounded
      by ``cutoff``; Voronoi neighbours with distances above ``cutoff`` are
      dropped).

    In both cases ``edge_weight`` is the Cartesian edge length, and the
    returned graph is bidirectional (see ``xanesnet.utils.graph``).
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
        # params
        cutoff: float,
        max_num_neighbors: int,
        compute_angles: bool,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
    ) -> None:
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.compute_angles = compute_angles
        self.graph_method = graph_method
        self.min_facet_area = min_facet_area
        self.cov_radii_scale = cov_radii_scale

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        pmg_obj = self.datasource[idx]
        for key in SPECTRUM_KEYS:
            if key in pmg_obj.site_properties.keys():
                break
        else:
            logging.warning(f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['file_name']}); skipping.")
            return 0

        xanes = np.array(pmg_obj.site_properties[key], dtype=object)
        xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()  # noqa: E711
        xanes = xanes[xanes_idxs]
        absorber_mask = torch.zeros(len(pmg_obj.labels), dtype=torch.bool)
        absorber_mask[xanes_idxs] = True
        intensities_np = np.array([x["intensities"] for x in xanes], dtype=np.float32)
        energies_np = np.array([x["energies"] for x in xanes], dtype=np.float32)

        atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
        cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)
        energies = torch.tensor(energies_np, dtype=torch.float32)
        intensities = torch.tensor(intensities_np, dtype=torch.float32)

        edge_index, edge_weight, angle, idx_kj, idx_ji = self._build_edges(
            pmg_obj,
            self.cutoff,
            self.max_num_neighbors,
            self.compute_angles,
            self.graph_method,
            self.min_facet_area,
            self.cov_radii_scale,
        )

        struct = GeometryGraphData(
            x=atomic_numbers,
            pos=cart_coords,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=None,
            angle=angle,
            idx_kj=idx_kj,
            idx_ji=idx_ji,
            energies=energies,
            intensities=intensities,
            absorber_mask=absorber_mask,
            file_name=pmg_obj.properties["file_name"],
        )

        self._save_data(struct, save_path_fn(0))
        return 1

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
        compute_angles: bool,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Build edges (and optionally triplet angles) for the molecular/crystal graph using the selected ``graph_method``.

        When compute_angles is False, returns:
            (edge_index, edge_weight, None, None, None)
        When compute_angles is True, returns:
            (edge_index, edge_weight, angle, idx_kj, idx_ji)
        where angle, idx_kj, idx_ji correspond to triplets (k->j->i) as used by e.g. DimeNet.
        """
        edge_index, edge_weight, edge_vec, _edge_attr = build_edges(
            pmg_obj,
            cutoff,
            max_num_neighbors,
            compute_vectors=compute_angles,
            method=graph_method,
            min_facet_area=min_facet_area,
            cov_radii_scale=cov_radii_scale,
        )

        if not compute_angles:
            return edge_index, edge_weight, None, None, None

        assert edge_vec is not None
        is_periodic = isinstance(pmg_obj, Structure)
        angle, idx_kj, idx_ji = compute_triplets_and_angles(
            edge_index, edge_vec, num_nodes=len(pmg_obj), is_periodic=is_periodic
        )
        return edge_index, edge_weight, angle, idx_kj, idx_ji

    @staticmethod
    def _save_data(data: Data, path: str) -> None:
        tensor_dict = data.to_dict()
        torch.save(tensor_dict, path)

    def _load_item(self, path: str) -> GeometryGraphData:
        tensor_dict = torch.load(path, weights_only=True)
        return GeometryGraphData(**tensor_dict)

    @property
    def signature(self) -> Config:
        """
        Return dataset signature as a dictionary.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "cutoff": self.cutoff,
                "max_num_neighbors": self.max_num_neighbors,
                "compute_angles": self.compute_angles,
                "graph_method": self.graph_method,
                "min_facet_area": self.min_facet_area,
                "cov_radii_scale": self.cov_radii_scale,
            }
        )
        return signature
