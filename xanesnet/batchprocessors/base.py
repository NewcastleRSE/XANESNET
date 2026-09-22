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

"""Abstract base class for all XANESNET batch processors."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from xanesnet.datasets import Dataset
    from xanesnet.encodings import SpectraEncoding


class BatchProcessor(ABC):
    """Abstract base class for batch processors.

    Converts a dataset batch into model inputs and targets for a specific
    model architecture. Subclasses must implement the data-shaping methods
    :meth:`input_preparation`, :meth:`target_preparation`, and
    :meth:`sample_id_preparation`, plus the three encoding hooks
    :meth:`encode_input`, :meth:`encode_target`, and :meth:`decode_target`.

    **Spectra encoding.** The stored
    :class:`~xanesnet.encodings.SpectraEncoding` is applied to spectra only.
    Whether the spectrum is the model *input* or the model *target* depends on
    the prediction direction, so the encoding is applied through three hooks:

    * :meth:`encode_input` -- map raw model inputs into the space the model
      consumes (encodes the spectral input for inverse prediction).
    * :meth:`encode_target` -- map raw targets into the model's prediction
      space before the loss (encodes the spectral target for forward
      prediction).
    * :meth:`decode_target` -- map predictions (in the model's prediction
      space) back to the raw target space during inference (decodes spectra
      for forward prediction).

    These hooks are abstract here to make the contract explicit. Concrete
    processors normally inherit them from one of the two direction-specific
    parents,
    :class:`~xanesnet.batchprocessors.forward.base.ForwardBatchProcessor`
    (spectra are the target) or
    :class:`~xanesnet.batchprocessors.inverse.base.InverseBatchProcessor`
    (spectra are the input), and only implement the data-shaping methods.

    :meth:`input_preparation` and :meth:`target_preparation` always return
    raw tensors so that inferencers can store ground truth in the original
    space and auto-config can resolve raw dimensions. When ``encoding`` is
    ``None`` (e.g. during auto-config resolution) all encode/decode hooks
    become pass-throughs.

    Args:
        encoding: Optional spectra encoding. When provided, the direction
            parent applies it in its hooks. When ``None``, all encode/decode
            hooks return their inputs unchanged.
    """

    def __init__(self, encoding: "SpectraEncoding | None" = None) -> None:
        self._encoding = encoding

    @abstractmethod
    def input_preparation(self, batch: Any) -> dict[str, Any]:
        """Prepare model input tensors from a batch.

        Args:
            batch: A collated batch produced by the dataset's ``collate_fn``.

        Returns:
            Dict mapping argument names to tensors (or related objects)
            expected by the model's ``forward`` method.
        """
        ...

    def input_preparation_single(self, dataset: "Dataset", index: int) -> dict[str, Any]:
        """Prepare model inputs from a single dataset sample.

        Collates the sample at ``index`` into a batch of size 1 and delegates
        to :meth:`input_preparation`.

        Args:
            dataset: The dataset to draw the sample from.
            index: Index of the sample within the dataset.

        Returns:
            Dict mapping argument names to tensors expected by the model.
        """
        sample = dataset[index]
        batch = dataset.collate_fn([sample])
        return self.input_preparation(batch)

    def prediction_preparation(self, batch: Any, predictions: torch.Tensor) -> torch.Tensor:
        """Post-process raw model predictions before loss computation.

        The default implementation returns predictions unchanged. Override in subclasses
        to apply masking or other per-batch transformations (e.g. selecting target sites
        from a per-atom output tensor).

        Args:
            batch: The collated batch (may carry masks or indices needed for post-processing).
            predictions: Raw model output tensor.

        Returns:
            Post-processed predictions tensor.
        """
        return predictions

    @abstractmethod
    def target_preparation(self, batch: Any) -> torch.Tensor:
        """Prepare the target tensor from a batch.

        Args:
            batch: A collated batch produced by the dataset's ``collate_fn``.

        Returns:
            Target tensor for loss computation.
        """
        ...

    def target_preparation_single(self, dataset: "Dataset", index: int) -> torch.Tensor:
        """Prepare the target tensor from a single dataset sample.

        Collates the sample at ``index`` into a batch of size 1 and delegates
        to :meth:`target_preparation`.

        Args:
            dataset: The dataset to draw the sample from.
            index: Index of the sample within the dataset.

        Returns:
            Target tensor for loss computation.
        """
        sample = dataset[index]
        batch = dataset.collate_fn([sample])
        return self.target_preparation(batch)

    def element_preparation(self, batch: Any) -> torch.Tensor | None:
        """Prepare per-target-site atomic numbers from a batch.

        The returned tensor is aligned row-wise with :meth:`target_preparation`,
        carrying the target-site element's atomic number for each target spectrum.
        The default implementation returns ``None``, indicating that no element
        information is available for this dataset/model combination.

        Args:
            batch: A collated batch produced by the dataset's ``collate_fn``.

        Returns:
            Per-target atomic numbers ``(batch_size,)``, or ``None`` when the
            batch carries no element information.
        """
        return None

    def element_preparation_single(self, dataset: "Dataset", index: int) -> torch.Tensor | None:
        """Prepare target-site atomic numbers from a single dataset sample.

        Collates the sample at ``index`` into a batch of size 1 and delegates
        to :meth:`element_preparation`.

        Args:
            dataset: The dataset to draw the sample from.
            index: Index of the sample within the dataset.

        Returns:
            Per-target atomic numbers for the sample, or ``None`` when the
            batch carries no element information.
        """
        sample = dataset[index]
        batch = dataset.collate_fn([sample])
        return self.element_preparation(batch)

    def target_site_index_preparation(self, batch: Any) -> torch.Tensor | None:
        """Prepare original raw-structure atom indices for target-sites.

        The returned tensor is aligned row-wise with :meth:`target_preparation`.
        Each value indexes the raw structure identified by the corresponding
        ``sample_id``. ``None`` denotes predictions that are not specific to
        an atom in the raw structure.

        Args:
            batch: A collated batch produced by the dataset's ``collate_fn``.

        Returns:
            Original target-site atom indices ``(n_target_sites,)``, or
            ``None`` for non-site-specific predictions.
        """
        return getattr(batch, "target_site_index", None)

    @abstractmethod
    def encode_input(self, inputs: dict[str, Any], elements: torch.Tensor | None = None) -> dict[str, Any]:
        """Map raw model inputs into the space the model consumes.

        Called by runners after :meth:`input_preparation`. Forward processors
        return the inputs unchanged; inverse processors encode the spectral
        input in place.

        Args:
            inputs: Input dict returned by :meth:`input_preparation`.
            elements: Optional per-sample target-site atomic numbers ``(B,)``
                forwarded to element-aware encodings.

        Returns:
            Input dict ready for ``model.forward``.
        """
        ...

    @abstractmethod
    def encode_target(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Map raw targets into the model's prediction space.

        Called by trainers before the loss is computed. Forward processors
        encode the spectral target; inverse processors return the descriptor
        target unchanged.

        Args:
            targets: Raw target tensor from :meth:`target_preparation`.
            elements: Optional per-sample target-site atomic numbers ``(B,)``
                forwarded to element-aware encodings.

        Returns:
            Target tensor in the model's prediction space.
        """
        ...

    @abstractmethod
    def decode_target(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Map predictions from the model's prediction space to raw target space.

        Called by inferencers after :meth:`prediction_preparation`. Forward
        processors decode the spectral prediction; inverse processors return
        the descriptor prediction unchanged. Acts as the inverse of
        :meth:`encode_target`.

        Args:
            predictions: Model output tensor after
                :meth:`prediction_preparation`.
            elements: Optional per-sample target-site atomic numbers ``(B,)``
                forwarded to element-aware decodings.

        Returns:
            Predictions in the raw target space.
        """
        ...

    @abstractmethod
    def sample_id_preparation(self, batch: Any) -> np.ndarray:
        """Prepare file name identifiers from a batch.

        Args:
            batch: A collated batch produced by the dataset's ``collate_fn``.

        Returns:
            Array of file name strings. ``(batch_size,)``
        """
        ...
