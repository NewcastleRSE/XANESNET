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
from typing import Any

import numpy as np
import torch

from xanesnet.datasets._mp import mp_save_path, run_mp_prepare
from xanesnet.datasets.base import Dataset as _BaseDataset
from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config
from xanesnet.utils.graph import build_edges
from xanesnet.utils.graph.gemnet_indices import (
    compute_id_swap,
    compute_mixed_triplets,
    compute_quadruplets,
    compute_triplets,
)

from ...registry import DatasetRegistry
from ..gemnet import SPECTRUM_KEYS, GemNetData, GemNetDataset


@DatasetRegistry.register("gemnet_mp")
@DatasetRegistry.register("gemnet_oc_mp")
class GemNetDatasetMp(GemNetDataset):
    """Multiprocessing variant of :class:`GemNetDataset` (covers GemNet and GemNet-OC)."""

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
        skip_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
        cutoff: float,
        max_num_neighbors: int,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
        quadruplets: bool,
        int_cutoff: float | None,
        int_max_neighbors: int | None = None,
        int_graph_method: str | None = None,
        int_min_facet_area: float | str | None = None,
        int_cov_radii_scale: float | None = None,
        oc_mode: bool = False,
        oc_cutoff_aeaint: float | None = None,
        oc_cutoff_aint: float | None = None,
        oc_max_neighbors_aeaint: int | None = None,
        oc_max_neighbors_aint: int | None = None,
        oc_graph_method_aeaint: str | None = None,
        oc_min_facet_area_aeaint: float | str | None = None,
        oc_cov_radii_scale_aeaint: float | None = None,
        oc_graph_method_aint: str | None = None,
        oc_min_facet_area_aint: float | str | None = None,
        oc_cov_radii_scale_aint: float | None = None,
        num_workers: int | None = None,
    ) -> None:
        super().__init__(
            dataset_type=dataset_type,
            datasource=datasource,
            root=root,
            preload=preload,
            skip_prepare=skip_prepare,
            split_ratios=split_ratios,
            split_indexfile=split_indexfile,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            graph_method=graph_method,
            min_facet_area=min_facet_area,
            cov_radii_scale=cov_radii_scale,
            quadruplets=quadruplets,
            int_cutoff=int_cutoff,
            int_max_neighbors=int_max_neighbors,
            int_graph_method=int_graph_method,
            int_min_facet_area=int_min_facet_area,
            int_cov_radii_scale=int_cov_radii_scale,
            oc_mode=oc_mode,
            oc_cutoff_aeaint=oc_cutoff_aeaint,
            oc_cutoff_aint=oc_cutoff_aint,
            oc_max_neighbors_aeaint=oc_max_neighbors_aeaint,
            oc_max_neighbors_aint=oc_max_neighbors_aint,
            oc_graph_method_aeaint=oc_graph_method_aeaint,
            oc_min_facet_area_aeaint=oc_min_facet_area_aeaint,
            oc_cov_radii_scale_aeaint=oc_cov_radii_scale_aeaint,
            oc_graph_method_aint=oc_graph_method_aint,
            oc_min_facet_area_aint=oc_min_facet_area_aint,
            oc_cov_radii_scale_aint=oc_cov_radii_scale_aint,
        )
        self.num_workers = num_workers

    def prepare(self) -> bool:
        skip_processing = _BaseDataset.prepare(self)
        if skip_processing:
            return True

        self._length = run_mp_prepare(self, len(self.datasource), self.num_workers)
        return True

    def _process_range(self, start: int, end: int) -> None:
        for idx in range(start, end):
            pmg_obj = self.datasource[idx]
            for key in SPECTRUM_KEYS:
                if key in pmg_obj.site_properties.keys():
                    break
            else:
                logging.warning(
                    f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['file_name']}); skipping."
                )
                continue

            xanes = np.array(pmg_obj.site_properties[key], dtype=object)
            xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()  # noqa: E711
            if len(xanes_idxs) == 0:
                logging.warning(f"No absorbers for sample {idx}; skipping.")
                continue

            xanes = xanes[xanes_idxs]
            intensities = np.stack([x["intensities"] for x in xanes]).astype(np.float32)
            energies = np.stack([x["energies"] for x in xanes]).astype(np.float32)

            n_atoms = len(pmg_obj.atomic_numbers)
            absorber_mask = torch.zeros(n_atoms, dtype=torch.bool)
            absorber_mask[xanes_idxs] = True

            atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
            cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)

            main_params = (
                self.cutoff,
                self.max_num_neighbors,
                self.graph_method,
                self.min_facet_area,
                self.cov_radii_scale,
            )
            edge_index, edge_weight, edge_vec, _ = build_edges(
                pmg_obj,
                self.cutoff,
                self.max_num_neighbors,
                compute_vectors=True,
                method=self.graph_method,
                min_facet_area=self.min_facet_area,
                cov_radii_scale=self.cov_radii_scale,
            )
            assert edge_vec is not None

            if edge_index.size(1) > 0:
                id_swap = compute_id_swap(edge_index, edge_vec)
                id3_reduce_ca, id3_expand_ba, Kidx3 = compute_triplets(edge_index, n_atoms)
            else:
                id_swap = torch.empty(0, dtype=torch.int64)
                id3_reduce_ca = torch.empty(0, dtype=torch.int64)
                id3_expand_ba = torch.empty(0, dtype=torch.int64)
                Kidx3 = torch.empty(0, dtype=torch.int64)

            data_fields: dict[str, Any] = {
                "x": atomic_numbers,
                "pos": cart_coords,
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "edge_vec": edge_vec,
                "id_c": edge_index[0].clone().to(torch.int64),
                "id_a": edge_index[1].clone().to(torch.int64),
                "id_swap": id_swap,
                "id3_reduce_ca": id3_reduce_ca,
                "id3_expand_ba": id3_expand_ba,
                "Kidx3": Kidx3,
                "energies": torch.tensor(energies, dtype=torch.float32),
                "intensities": torch.tensor(intensities, dtype=torch.float32),
                "absorber_mask": absorber_mask,
                "file_name": pmg_obj.properties["file_name"],
            }

            int_edge_index: torch.Tensor | None = None
            int_edge_weight: torch.Tensor | None = None
            int_edge_vec: torch.Tensor | None = None
            int_params = (
                self.int_cutoff,
                self.int_max_neighbors,
                self.int_graph_method,
                self.int_min_facet_area,
                self.int_cov_radii_scale,
            )
            if self.quadruplets:
                if int_params == main_params:
                    int_edge_index = edge_index
                    int_edge_weight = edge_weight
                    int_edge_vec = edge_vec
                else:
                    int_edge_index, int_edge_weight, int_edge_vec, _ = build_edges(
                        pmg_obj,
                        self.int_cutoff,
                        self.int_max_neighbors,
                        compute_vectors=True,
                        method=self.int_graph_method,
                        min_facet_area=self.int_min_facet_area,
                        cov_radii_scale=self.int_cov_radii_scale,
                    )
                assert int_edge_vec is not None
                data_fields.update(
                    {
                        "int_edge_index": int_edge_index,
                        "int_edge_weight": int_edge_weight,
                        "int_edge_vec": int_edge_vec,
                        "id4_int_b": int_edge_index[0].clone().to(torch.int64),
                        "id4_int_a": int_edge_index[1].clone().to(torch.int64),
                    }
                )
                if int_edge_index.size(1) > 0 and edge_index.size(1) > 0:
                    quad = compute_quadruplets(edge_index, edge_vec, int_edge_index, int_edge_vec, n_atoms)
                else:
                    empty = torch.empty(0, dtype=torch.int64)
                    quad = dict(
                        id4_reduce_ca=empty,
                        id4_expand_db=empty,
                        id4_reduce_cab=empty,
                        id4_expand_abd=empty,
                        id4_reduce_intm_ca=empty,
                        id4_expand_intm_db=empty,
                        id4_reduce_intm_ab=empty,
                        id4_expand_intm_ab=empty,
                        Kidx4=empty,
                    )
                data_fields.update(quad)

            if self.oc_mode:
                aeaint_params = (
                    self.oc_cutoff_aeaint,
                    self.oc_max_neighbors_aeaint,
                    self.oc_graph_method_aeaint,
                    self.oc_min_facet_area_aeaint,
                    self.oc_cov_radii_scale_aeaint,
                )
                if aeaint_params == main_params:
                    a2ee2a_edge_index = edge_index
                    a2ee2a_edge_weight = edge_weight
                    a2ee2a_edge_vec = edge_vec
                else:
                    a2ee2a_edge_index, a2ee2a_edge_weight, a2ee2a_edge_vec, _ = build_edges(
                        pmg_obj,
                        self.oc_cutoff_aeaint,
                        self.oc_max_neighbors_aeaint,
                        compute_vectors=True,
                        method=self.oc_graph_method_aeaint,
                        min_facet_area=self.oc_min_facet_area_aeaint,
                        cov_radii_scale=self.oc_cov_radii_scale_aeaint,
                    )
                aint_params = (
                    self.oc_cutoff_aint,
                    self.oc_max_neighbors_aint,
                    self.oc_graph_method_aint,
                    self.oc_min_facet_area_aint,
                    self.oc_cov_radii_scale_aint,
                )
                if self.quadruplets and aint_params == int_params:
                    assert int_edge_index is not None and int_edge_weight is not None and int_edge_vec is not None
                    a2a_edge_index = int_edge_index
                    a2a_edge_weight = int_edge_weight
                    a2a_edge_vec = int_edge_vec
                elif aint_params == main_params:
                    a2a_edge_index = edge_index
                    a2a_edge_weight = edge_weight
                    a2a_edge_vec = edge_vec
                else:
                    a2a_edge_index, a2a_edge_weight, a2a_edge_vec, _ = build_edges(
                        pmg_obj,
                        self.oc_cutoff_aint,
                        self.oc_max_neighbors_aint,
                        compute_vectors=True,
                        method=self.oc_graph_method_aint,
                        min_facet_area=self.oc_min_facet_area_aint,
                        cov_radii_scale=self.oc_cov_radii_scale_aint,
                    )
                assert a2ee2a_edge_vec is not None and a2a_edge_vec is not None

                data_fields.update(
                    {
                        "a2ee2a_edge_index": a2ee2a_edge_index,
                        "a2ee2a_edge_weight": a2ee2a_edge_weight,
                        "a2ee2a_edge_vec": a2ee2a_edge_vec,
                        "a2a_edge_index": a2a_edge_index,
                        "a2a_edge_weight": a2a_edge_weight,
                        "a2a_edge_vec": a2a_edge_vec,
                    }
                )

                if self.quadruplets:
                    data_fields["qint_edge_index"] = data_fields["int_edge_index"]
                    data_fields["qint_edge_weight"] = data_fields["int_edge_weight"]
                    data_fields["qint_edge_vec"] = data_fields["int_edge_vec"]
                else:
                    qint_edge_index, qint_edge_weight, qint_edge_vec, _ = build_edges(
                        pmg_obj,
                        self.int_cutoff,
                        self.int_max_neighbors,
                        compute_vectors=True,
                        method=self.int_graph_method,
                        min_facet_area=self.int_min_facet_area,
                        cov_radii_scale=self.int_cov_radii_scale,
                    )
                    assert qint_edge_vec is not None
                    data_fields["qint_edge_index"] = qint_edge_index
                    data_fields["qint_edge_weight"] = qint_edge_weight
                    data_fields["qint_edge_vec"] = qint_edge_vec

                data_fields["trip_e2e_in"] = data_fields["id3_expand_ba"]
                data_fields["trip_e2e_out"] = data_fields["id3_reduce_ca"]
                data_fields["trip_e2e_out_agg"] = data_fields["Kidx3"]

                a2e = compute_mixed_triplets(
                    main_edge_index=edge_index,
                    main_edge_vec=edge_vec,
                    other_edge_index=a2ee2a_edge_index,
                    other_edge_vec=a2ee2a_edge_vec,
                    num_nodes=n_atoms,
                    to_outedge=False,
                )
                e2a = compute_mixed_triplets(
                    main_edge_index=a2ee2a_edge_index,
                    main_edge_vec=a2ee2a_edge_vec,
                    other_edge_index=edge_index,
                    other_edge_vec=edge_vec,
                    num_nodes=n_atoms,
                    to_outedge=False,
                )
                data_fields.update(
                    {
                        "trip_a2e_in": a2e["in_"],
                        "trip_a2e_out": a2e["out"],
                        "trip_a2e_out_agg": a2e["out_agg"],
                        "trip_e2a_in": e2a["in_"],
                        "trip_e2a_out": e2a["out"],
                        "trip_e2a_out_agg": e2a["out_agg"],
                    }
                )

            struct = GemNetData(**data_fields)
            self._save_data(struct, mp_save_path(self.processed_dir, idx, 0))

    @property
    def signature(self) -> Config:
        return super().signature
