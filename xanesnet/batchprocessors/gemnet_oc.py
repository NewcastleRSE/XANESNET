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

import numpy as np
import torch

from xanesnet.datasets import GemNetBatch

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("gemnet_oc", "gemnet_oc")
class GemNetOCBatchProcessor(BatchProcessor):
    """
    Batch processor for the PyG-based ``GemNetDataset`` feeding GemNet-OC.

    GemNet-OC reads a large set of precomputed tensors (main/qint/a2ee2a/a2a
    graphs plus triplet, mixed-triplet and quadruplet indices). All are
    produced offline by ``GemNetDataset.prepare()`` and forwarded here as
    individual ``torch.Tensor`` kwargs (consistent with the GemNet processor).
    Optional fields are looked up with ``getattr(batch, name, None)`` so that
    downstream toggles on the model side can gate their use.
    """

    _OPTIONAL_KEYS: tuple[str, ...] = (
        # Quadruplet graph / indices
        "qint_edge_index",
        "qint_edge_weight",
        "qint_edge_vec",
        "id4_expand_intm_db",
        "id4_expand_intm_ab",
        "id4_reduce_intm_ab",
        "id4_reduce_intm_ca",
        "id4_reduce_ca",
        "id4_expand_abd",
        "id4_reduce_cab",
        "Kidx4",
        # a2ee2a graph
        "a2ee2a_edge_index",
        "a2ee2a_edge_weight",
        "a2ee2a_edge_vec",
        # Mixed triplets
        "trip_a2e_in",
        "trip_a2e_out",
        "trip_a2e_out_agg",
        "trip_e2a_in",
        "trip_e2a_out",
        "trip_e2a_out_agg",
        # a2a graph
        "a2a_edge_index",
        "a2a_edge_weight",
        "a2a_edge_vec",
    )

    def input_preparation(self, batch: GemNetBatch) -> dict[str, torch.Tensor | None]:
        inputs: dict[str, torch.Tensor | None] = {
            "z": batch.x,
            "edge_index": batch.edge_index,
            "edge_weight": batch.edge_weight,
            "edge_vec": batch.edge_vec,
            "id_swap": batch.id_swap,
            "id3_expand_ba": batch.id3_expand_ba,
            "id3_reduce_ca": batch.id3_reduce_ca,
            "Kidx3": batch.Kidx3,
        }
        for key in self._OPTIONAL_KEYS:
            inputs[key] = getattr(batch, key, None)
        return inputs

    def prediction_preparation(self, batch: GemNetBatch, predictions: torch.Tensor) -> torch.Tensor:
        return predictions[batch.absorber_mask]

    def target_preparation(self, batch: GemNetBatch) -> torch.Tensor:
        return batch.intensities

    def file_name_extraction(self, batch: GemNetBatch) -> np.ndarray:
        return np.array(batch.file_name, dtype=str)
