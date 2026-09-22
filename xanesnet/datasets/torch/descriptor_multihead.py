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

"""Descriptor-based tensor dataset for multi-head models."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from xanesnet.datasources import DataSource
from xanesnet.descriptors import Descriptor, DescriptorRegistry
from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError

from ..base import SavePathFn, TorchDataset
from ..registry import DatasetRegistry


@dataclass
class DescriptorMultiheadData:
    """Container for one descriptor-based multi-head sample or batch.

    Attributes:
        x: Model input tensor. In forward mode this is a descriptor tensor;
            in inverse mode it is a spectrum tensor.
        y: Model target tensor. In forward mode this is a spectrum tensor;
            in inverse mode it is a descriptor tensor.
        energies: Energy grid tensor with shape ``(n_energies,)`` or ``(batch, n_energies)``.
        sample_id: Sample identifier metadata for one sample or a batch.
        element: Absorber atomic number as a scalar tensor for one sample, or
            ``(batch,)`` for a batch. Consumed by element-aware spectra
            encodings.
        head_idx: Active multi-head index for one sample, or ``(batch,)`` for
            a batch. Populated from ``subdir_id`` on the datasource entry.
    """

    x: torch.Tensor | None = None
    y: torch.Tensor | None = None
    energies: torch.Tensor | None = None
    sample_id: str | list[Any] | None = None
    element: torch.Tensor | None = None
    head_idx: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "DescriptorMultiheadData":
        """Move tensor attributes to ``device`` in place.

        Args:
            device: Target device accepted by ``torch.Tensor.to``.

        Returns:
            This data object after moving tensor attributes.
        """
        for attr in ["x", "y", "energies", "element", "head_idx"]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize this sample to a torch-saveable state dictionary.

        Returns:
            Dictionary containing tensor and metadata fields.
        """
        return {
            "x": self.x,
            "y": self.y,
            "energies": self.energies,
            "sample_id": self.sample_id,
            "element": self.element,
            "head_idx": self.head_idx,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DescriptorMultiheadData":
        """Create data from a state dictionary.

        Args:
            state: State dictionary produced by ``to_state_dict``.

        Returns:
            Reconstructed multi-head data object.
        """
        return cls(
            x=state.get("x"),
            y=state.get("y"),
            energies=state.get("energies"),
            sample_id=state.get("sample_id"),
            element=state.get("element"),
            head_idx=state.get("head_idx"),
        )

    def save(self, path: str) -> str:
        """Save this data object to disk.

        Args:
            path: Destination ``.pth`` path.

        Returns:
            The destination path.
        """
        torch.save(self.to_state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str) -> "DescriptorMultiheadData":
        """Load multi-head data from disk.

        Args:
            path: Source ``.pth`` path.

        Returns:
            Loaded multi-head data object.
        """
        state = torch.load(path, weights_only=True)
        return cls.from_state_dict(state)


@DatasetRegistry.register("descriptor_multihead")
@DatasetRegistry.register("descriptor_multihead_inverse")
class DescriptorMultiheadDataset(TorchDataset):
    """Descriptor-based dataset for multi-head models.

    Like :class:`~xanesnet.datasets.torch.descriptor.DescriptorDataset`, this
    dataset computes the configured structural descriptors for each target
    site. It additionally stores ``head_idx`` from the datasource
    ``subdir_id`` property so batch processors can select the active prediction
    head. The dataset type controls the direction: forward types map
    descriptors to spectra, while inverse types map spectra to descriptors.

    Args:
        dataset_type: Registered dataset type name. Canonical names are
            ``"descriptor_multihead"``, ``"descriptor_multihead_mp"``,
            ``"descriptor_multihead_inverse"``, and
            ``"descriptor_multihead_inverse_mp"``.
        datasource: Raw datasource of pymatgen structures or molecules.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        descriptors: Descriptor configuration objects.

    Raises:
        ConfigError: If a datasource entry does not provide ``subdir_id``.
    """

    _INVERSE_MARKER = "_inverse"

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
        descriptors: list[Config],
    ) -> None:
        """Initialize the descriptor-based multi-head dataset."""
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self._inverse = self._INVERSE_MARKER in dataset_type
        self._num_heads: int | None = None

        self.descriptor_configs = descriptors
        self.descriptor_list: list[Descriptor] = []
        descriptor_types = ", ".join(d.get_str("descriptor_type") for d in descriptors)
        logging.info(f"Initializing descriptors: {descriptor_types}")
        for descriptor_config in descriptors:
            descriptor_type = descriptor_config.get_str("descriptor_type")
            descriptor = DescriptorRegistry.create(descriptor_type, **descriptor_config.as_kwargs())
            self.descriptor_list.append(descriptor)

    @property
    def num_heads(self) -> int:
        """Return the number of contiguous prediction heads in the dataset.

        Head indices are assigned from the datasource subdirectories in sorted
        subdirectory-name order and are expected to be zero-based and
        contiguous. The value is cached after the first lookup because
        automatic model configuration may request it more than once during a
        run.

        Returns:
            Number of prediction heads.

        Raises:
            ConfigError: If samples do not provide valid contiguous head
                indices or if the dataset is empty.
        """
        if self._num_heads is not None:
            return self._num_heads

        head_indices: set[int] = set()
        for idx in range(len(self)):
            head_idx = self[idx].head_idx
            if head_idx is None or head_idx.ndim != 0:
                raise ConfigError("Descriptor multi-head samples must provide scalar head_idx values.")
            head_indices.add(int(head_idx.item()))

        if not head_indices:
            raise ConfigError("Cannot determine multi-head count from an empty dataset.")

        expected_head_indices = set(range(max(head_indices) + 1))
        if head_indices != expected_head_indices:
            raise ConfigError(
                "Descriptor multi-head samples must use contiguous zero-based head_idx values; "
                f"found {sorted(head_indices)}."
            )

        self._num_heads = len(head_indices)
        return self._num_heads

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into descriptor multi-head samples.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps per-item sample sequence numbers to output paths.

        Returns:
            Number of processed absorber samples written.
        """
        pmg_obj = self.datasource[idx]
        if "spectrum" not in pmg_obj.site_properties:
            logging.warning(f"No spectrum found for sample {idx} ({pmg_obj.properties['sample_id']}); skipping.")
            return 0

        spectra = np.array(pmg_obj.site_properties["spectrum"], dtype=object)
        target_site_indices: list[int] = np.where(spectra != None)[0].tolist()

        descriptor_features = []
        for descriptor in self.descriptor_list:
            feature = descriptor.transform_pmg(pmg_obj, site_index=target_site_indices)
            descriptor_features.append(feature)
        descriptor_features = np.concatenate(descriptor_features, axis=1)

        if "subdir_id" not in pmg_obj.properties:
            raise ConfigError(
                "Descriptor multi-head datasource entries must provide ``subdir_id`` for head assignment."
            )
        head_idx = torch.tensor(pmg_obj.properties["subdir_id"], dtype=torch.int64)

        seq = 0
        for site_idx, df in zip(target_site_indices, descriptor_features):
            df = torch.tensor(df, dtype=torch.float32)
            element = torch.tensor(pmg_obj.atomic_numbers[site_idx], dtype=torch.int64)

            spectrum = pmg_obj.site_properties["spectrum"][site_idx]
            energies = torch.tensor(spectrum["energies"], dtype=torch.float32)
            intensities = torch.tensor(spectrum["intensities"], dtype=torch.float32)

            if self._inverse:
                x = intensities
                y = df
            else:
                x = df
                y = intensities

            data = DescriptorMultiheadData(
                x=x,
                y=y,
                energies=energies,
                sample_id=pmg_obj.properties["sample_id"],
                element=element,
                head_idx=head_idx,
            )
            data.save(save_path_fn(seq))
            seq += 1

        return seq

    def collate_fn(self, batch: list[DescriptorMultiheadData]) -> DescriptorMultiheadData:
        """Collate multi-head samples into a batch.

        Args:
            batch: Multi-head samples loaded by ``__getitem__``.

        Returns:
            Batched multi-head data with stacked tensor fields.
        """

        def _stack(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
            """Stack tensors unless any field is absent for the batch."""
            if any(t is None for t in tensors):
                return None
            return torch.stack([tensor for tensor in tensors if tensor is not None])

        return DescriptorMultiheadData(
            x=_stack([b.x for b in batch]),
            y=_stack([b.y for b in batch]),
            energies=_stack([b.energies for b in batch]),
            sample_id=[b.sample_id for b in batch],
            element=_stack([b.element for b in batch]),
            head_idx=_stack([b.head_idx for b in batch]),
        )

    def _load_item(self, path: str) -> DescriptorMultiheadData:
        """Load one processed multi-head data sample.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Loaded multi-head data object.
        """
        return DescriptorMultiheadData.load(path)

    @property
    def signature(self) -> Config:
        """Return the dataset configuration signature.

        Returns:
            Configuration values that identify this dataset.
        """
        signature = super().signature
        signature.update_with_dict({"descriptors": self.descriptor_configs})
        return signature
