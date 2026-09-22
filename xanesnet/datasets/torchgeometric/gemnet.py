# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   ...

"""GemNet and GemNet-OC PyTorch Geometric dataset implementation."""

import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from xanesnet.datasources import DataSource
from xanesnet.graphs import GraphBuilder, GraphBuilderRegistry
from xanesnet.graphs.utils.directional_indices import (
    compute_id_swap,
    compute_mixed_triplets,
    compute_quadruplets,
    compute_triplets,
)
from xanesnet.serialization.config import Config

from ..base import SavePathFn, TorchGeometricDataset
from ..registry import DatasetRegistry


class GemNetData(Data):
    """PyG ``Data`` subclass for GemNet and GemNet-OC samples.

    The custom ``__inc__`` is critical for correct PyG batching of all edge /
    triplet / quadruplet / intermediate indices across multiple graphs.
    """

    # Node-level indices (offset by num_nodes when batched)
    _NODE_KEYS = {"id_c", "id_a", "id4_int_b", "id4_int_a"}
    # Main-graph edge-level indices (offset by num_main_edges)
    _MAIN_EDGE_KEYS = {
        "id_swap",
        "id3_expand_ba",
        "id3_reduce_ca",
        "id4_reduce_ca",
        "id4_expand_db",
        "id4_reduce_intm_ca",
        "id4_expand_intm_db",
        # OC mixed triplets where "out" edges are from the main graph
        "trip_e2e_in",
        "trip_e2e_out",
        "trip_a2e_out",
        "trip_e2a_in",
    }
    # Interaction-graph edge-level (offset by num_int_edges)
    _INT_EDGE_KEYS = {
        "id4_reduce_intm_ab",
        "id4_expand_intm_ab",
    }
    # a2ee2a-graph edge-level (offset by num_a2ee2a_edges)
    _A2EE2A_EDGE_KEYS = {
        "trip_a2e_in",
        "trip_e2a_out",
    }
    # Intermediate-ca level (offset by num_intm_ca)
    _INTM_CA_KEYS = {"id4_reduce_cab"}
    # Intermediate-db level (offset by num_intm_db)
    _INTM_DB_KEYS = {"id4_expand_abd"}
    # Ragged inner indices (no offset)
    _NO_INC_KEYS = {"Kidx3", "Kidx4", "trip_e2e_out_agg", "trip_a2e_out_agg", "trip_e2a_out_agg"}

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Return PyG batching increments for GemNet index fields.

        Args:
            key: Data attribute currently being batched.
            value: Attribute value currently being batched.
            *args: Additional PyG arguments.
            **kwargs: Additional PyG keyword arguments.

        Returns:
            Increment used by PyG for ``key``.
        """
        if (
            key in {"edge_index", "int_edge_index", "a2ee2a_edge_index", "a2a_edge_index", "qint_edge_index"}
            or key in self._NODE_KEYS
        ):
            return self.num_nodes
        if key in self._MAIN_EDGE_KEYS:
            assert self.edge_index is not None
            return self.edge_index.size(1)
        if key in self._INT_EDGE_KEYS:
            # Interaction edge graph is a 2xE tensor or an index into it
            return self.int_edge_index.size(1)
        if key in self._A2EE2A_EDGE_KEYS:
            return self.a2ee2a_edge_index.size(1)
        if key in self._INTM_CA_KEYS:
            return self.id4_reduce_intm_ca.size(0)
        if key in self._INTM_DB_KEYS:
            return self.id4_expand_intm_db.size(0)
        if key in self._NO_INC_KEYS:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the concatenation dimension for GemNet graph fields.

        Args:
            key: Data attribute currently being batched.
            value: Attribute value currently being batched.
            *args: Additional PyG arguments.
            **kwargs: Additional PyG keyword arguments.

        Returns:
            Concatenation dimension used by PyG for ``key``.
        """
        if key in {"edge_index", "int_edge_index", "a2ee2a_edge_index", "a2a_edge_index", "qint_edge_index"}:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class GemNetBatch(Batch):
    """Typed PyG batch emitted by ``GemNetDataset.collate_fn``.

    Calling :meth:`~torch_geometric.data.Batch.from_data_list` on this class
    preserves the concrete batch type while PyG adds its batching metadata.
    """

    x: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_vec: torch.Tensor
    id_c: torch.Tensor
    id_a: torch.Tensor
    id_swap: torch.Tensor
    id3_reduce_ca: torch.Tensor
    id3_expand_ba: torch.Tensor
    Kidx3: torch.Tensor
    target_site_mask: torch.Tensor
    target_site_index: torch.Tensor
    energies: torch.Tensor
    intensities: torch.Tensor
    sample_id: list[str]


@DatasetRegistry.register("gemnet")
@DatasetRegistry.register("gemnet_oc")
class GemNetDataset(TorchGeometricDataset):
    """Dataset for GemNet and GemNet-OC graph inputs.

    Mixing graph methods is supported (e.g. a Voronoi main graph paired with a
    radius-based ``int_graph_builder``). Whenever a sub-graph configuration is
    omitted (``None``) it defaults to the main ``graph_builder``. A derived
    graph is reused from a previously-built one only when the full builder
    configuration matches; otherwise it is built from scratch.

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw datasource of pymatgen structures or molecules.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        graph_builder: Main graph builder configuration.
        quadruplets: Whether to compute quadruplet indices.
        int_graph_builder: Interaction graph builder configuration. When
            ``None`` the main ``graph_builder`` is reused.
        oc_mode: Whether to precompute GemNet-OC auxiliary graphs and mixed
            triplets.
        oc_aeaint_graph_builder: Atom-edge-atom graph builder configuration.
            When ``None`` the main ``graph_builder`` is reused.
        oc_aint_graph_builder: Atom-atom graph builder configuration. When
            ``None`` the main ``graph_builder`` is reused.
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
        # params:
        graph_builder: Config,
        quadruplets: bool,
        int_graph_builder: Config | None = None,
        oc_mode: bool = False,
        oc_aeaint_graph_builder: Config | None = None,
        oc_aint_graph_builder: Config | None = None,
    ) -> None:
        """Initialize the GemNet dataset."""
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.graph_builder_config = graph_builder
        self.graph_builder = self._make_builder(graph_builder)
        self.quadruplets = quadruplets

        self.int_graph_builder_config = int_graph_builder if int_graph_builder is not None else graph_builder
        self.int_graph_builder = (
            self.graph_builder if int_graph_builder is None else self._make_builder(int_graph_builder)
        )

        self.oc_mode = oc_mode
        self.oc_aeaint_graph_builder_config = (
            oc_aeaint_graph_builder if oc_aeaint_graph_builder is not None else graph_builder
        )
        self.oc_aeaint_graph_builder = (
            self.graph_builder if oc_aeaint_graph_builder is None else self._make_builder(oc_aeaint_graph_builder)
        )
        self.oc_aint_graph_builder_config = (
            oc_aint_graph_builder if oc_aint_graph_builder is not None else graph_builder
        )
        self.oc_aint_graph_builder = (
            self.graph_builder if oc_aint_graph_builder is None else self._make_builder(oc_aint_graph_builder)
        )

        if oc_mode and not quadruplets:
            # GemNet-OC always needs quadruplet indices for its standard config;
            # allow disabling only if the user explicitly sets quadruplets=False.
            logging.info(
                "GemNetDataset: oc_mode=True without quadruplets=True. Quadruplet "
                "indices will NOT be computed; GemNet-OC's quad_interaction must be False."
            )

    @staticmethod
    def _make_builder(cfg: Config) -> GraphBuilder:
        """Instantiate a graph builder from a nested config.

        Args:
            cfg: Nested graph builder configuration.

        Returns:
            Concrete :class:`GraphBuilder` selected by ``graph_builder_type``.
        """
        return GraphBuilderRegistry.create(cfg.get_str("graph_builder_type"), **cfg.as_kwargs())

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into one GemNet graph sample.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps sample sequence numbers to output paths.

        Returns:
            ``1`` when the graph was saved, otherwise ``0`` when skipped.
        """
        pmg_obj = self.datasource[idx]
        if "spectrum" not in pmg_obj.site_properties:
            logging.warning(f"No spectrum found for sample {idx} ({pmg_obj.properties['sample_id']}); skipping.")
            return 0

        spectra = np.array(pmg_obj.site_properties["spectrum"], dtype=object)
        target_site_indices: list[int] = np.where(spectra != None)[0].tolist()  # noqa: E711
        if len(target_site_indices) == 0:
            logging.warning(f"No target sites for sample {idx}; skipping.")
            return 0

        spectra = spectra[target_site_indices]
        intensities = np.stack([x["intensities"] for x in spectra]).astype(np.float32)
        energies = np.stack([x["energies"] for x in spectra]).astype(np.float32)

        n_atoms = len(pmg_obj.atomic_numbers)
        target_site_mask = torch.zeros(n_atoms, dtype=torch.bool)
        target_site_mask[target_site_indices] = True

        atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
        cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)

        main_config_dict = self.graph_builder_config.as_dict()
        int_config_dict = self.int_graph_builder_config.as_dict()

        edge_index, edge_weight, edge_vec, _ = self.graph_builder.build(pmg_obj, compute_vectors=True)
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
            "target_site_mask": target_site_mask,
            "target_site_index": torch.tensor(target_site_indices, dtype=torch.int64),
            "sample_id": pmg_obj.properties["sample_id"],
        }

        int_edge_index = int_edge_weight = int_edge_vec = None
        if self.quadruplets:
            if int_config_dict == main_config_dict:
                int_edge_index = edge_index
                int_edge_weight = edge_weight
                int_edge_vec = edge_vec
            else:
                int_edge_index, int_edge_weight, int_edge_vec, _ = self.int_graph_builder.build(
                    pmg_obj, compute_vectors=True
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
            aeaint_config_dict = self.oc_aeaint_graph_builder_config.as_dict()
            if aeaint_config_dict == main_config_dict:
                a2ee2a_edge_index = edge_index
                a2ee2a_edge_weight = edge_weight
                a2ee2a_edge_vec = edge_vec
            else:
                a2ee2a_edge_index, a2ee2a_edge_weight, a2ee2a_edge_vec, _ = self.oc_aeaint_graph_builder.build(
                    pmg_obj, compute_vectors=True
                )
            aint_config_dict = self.oc_aint_graph_builder_config.as_dict()
            if self.quadruplets and aint_config_dict == int_config_dict:
                a2a_edge_index = int_edge_index
                a2a_edge_weight = int_edge_weight
                a2a_edge_vec = int_edge_vec
            elif aint_config_dict == main_config_dict:
                a2a_edge_index = edge_index
                a2a_edge_weight = edge_weight
                a2a_edge_vec = edge_vec
            else:
                a2a_edge_index, a2a_edge_weight, a2a_edge_vec, _ = self.oc_aint_graph_builder.build(
                    pmg_obj, compute_vectors=True
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
                qint_edge_index, qint_edge_weight, qint_edge_vec, _ = self.int_graph_builder.build(
                    pmg_obj, compute_vectors=True
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
        self._save_data(struct, save_path_fn(0))
        return 1

    def collate_fn(self, batch: list[BaseData]) -> GemNetBatch:
        """Collate GemNet graph samples into one PyG batch.

        Args:
            batch: GemNet graph samples loaded by ``__getitem__``.

        Returns:
            PyG batch with target tensors and file names concatenated over
            target sites.
        """
        fields_to_cat = ["energies", "intensities", "target_site_mask", "target_site_index"]
        batched = GemNetBatch.from_data_list(batch, exclude_keys=[*fields_to_cat, "sample_id"])
        for field in fields_to_cat:
            values = [getattr(data, field) for data in batch]
            setattr(batched, field, torch.cat(values, dim=0))
        batched.sample_id = [
            str(getattr(data, "sample_id"))
            for data in batch
            for _ in range(int(getattr(data, "target_site_mask").sum().item()))
        ]
        return batched

    @staticmethod
    def _save_data(data: GemNetData, path: str) -> None:
        """Save one GemNet data object as a tensor dictionary.

        Args:
            data: Data object to serialize.
            path: Destination ``.pth`` path.
        """
        torch.save(data.to_dict(), path)

    def _load_item(self, path: str) -> GemNetData:
        """Load one processed GemNet graph sample.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Reconstructed GemNet data object.
        """
        tensor_dict = torch.load(path, weights_only=True)
        return GemNetData(**tensor_dict)

    @property
    def signature(self) -> Config:
        """Dataset configuration signature.

        Returns:
            Configuration values that identify this GemNet dataset.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "graph_builder": self.graph_builder_config,
                "quadruplets": self.quadruplets,
                "int_graph_builder": self.int_graph_builder_config,
                "oc_mode": self.oc_mode,
                "oc_aeaint_graph_builder": self.oc_aeaint_graph_builder_config,
                "oc_aint_graph_builder": self.oc_aint_graph_builder_config,
            }
        )
        return signature
