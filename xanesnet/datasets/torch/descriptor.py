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

"""Descriptor-based tensor dataset implementation."""

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from xanesnet.datasources import DataSource
from xanesnet.descriptors import Descriptor, DescriptorRegistry
from xanesnet.serialization.config import Config

from ..base import SavePathFn, TorchDataset
from ..registry import DatasetRegistry


@dataclass
class DescriptorData:
    """Container for one descriptor dataset sample or batch.

    Attributes:
        x: Model input tensor, commonly ``(n_features,)`` or ``(batch, n_features)``.
        y: Model target tensor, commonly ``(n_energies,)`` or ``(batch, n_energies)``.
        energies: Energy grid tensor with shape ``(n_energies,)`` or ``(batch, n_energies)``.
        sample_id: Sample identifier metadata for one sample or a batch.
        element: Target-site atomic number as a scalar tensor for one sample, or
            ``(batch,)`` for a batch. Consumed by element-aware spectra
            encodings.
        target_site_index: Original target-site atom index as a scalar tensor
            for one sample, or ``(batch,)`` for a batch.
    """

    x: torch.Tensor | None = None
    y: torch.Tensor | None = None
    energies: torch.Tensor | None = None
    sample_id: str | list[Any] | None = None
    element: torch.Tensor | None = None
    target_site_index: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "DescriptorData":
        """Move tensor attributes to ``device`` in place.

        Args:
            device: Target device accepted by ``torch.Tensor.to``.

        Returns:
            This data object after moving tensor attributes.
        """
        for attr in ["x", "y", "energies", "element", "target_site_index"]:
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
            "target_site_index": self.target_site_index,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DescriptorData":
        """Create data from a state dictionary.

        Args:
            state: State dictionary produced by ``to_state_dict``.

        Returns:
            Reconstructed descriptor data object.
        """
        return cls(
            x=state["x"],
            y=state["y"],
            energies=state["energies"],
            sample_id=state["sample_id"],
            element=state["element"],
            target_site_index=state["target_site_index"],
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
    def load(cls, path: str) -> "DescriptorData":
        """Load descriptor data from disk.

        Args:
            path: Source ``.pth`` path.

        Returns:
            Loaded descriptor data object.
        """
        state = torch.load(path, weights_only=True)
        return cls.from_state_dict(state)


@DatasetRegistry.register("descriptor")
@DatasetRegistry.register("descriptor_inverse")
class DescriptorDataset(TorchDataset):
    """Dataset that converts structures to descriptor tensors.

    The prediction direction is detected from the dataset type: types
    containing ``"_inverse"`` (e.g. ``"descriptor_inverse"``,
    ``"descriptor_inverse_mp"``) swap inputs and targets so that spectra are
    the model input and structural descriptors are the prediction target.

    Args:
        dataset_type: Registered dataset type name (``"descriptor"`` or
            ``"descriptor_mp"`` for forward prediction; ``"descriptor_inverse"``
            or ``"descriptor_inverse_mp"`` for inverse prediction).
        datasource: Raw datasource of pymatgen structures or molecules.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        descriptors: Descriptor configuration objects.
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
        # descriptors
        descriptors: list[Config],
    ) -> None:
        """Initialize the descriptor dataset."""
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self._inverse = self._INVERSE_MARKER in dataset_type

        # Create descriptors
        self.descriptor_configs = descriptors
        self.descriptor_list: list[Descriptor] = []
        descriptor_types = ", ".join(d.get_str("descriptor_type") for d in descriptors)
        logging.info(f"Initializing descriptors: {descriptor_types}")
        for descriptor_config in descriptors:
            descriptor_type = descriptor_config.get_str("descriptor_type")
            descriptor = DescriptorRegistry.create(descriptor_type, **descriptor_config.as_kwargs())
            self.descriptor_list.append(descriptor)

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into descriptor samples.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps per-item sample sequence numbers to output paths.

        Returns:
            Number of processed target-site samples written.
        """
        pmg_obj = self.datasource[idx]
        if "spectrum" not in pmg_obj.site_properties:
            logging.warning(f"No spectrum found for sample {idx} ({pmg_obj.properties['sample_id']}); skipping.")
            return 0

        spectra = np.array(pmg_obj.site_properties["spectrum"], dtype=object)
        target_site_indices: list[int] = np.where(spectra != None)[0].tolist()

        # Compute descriptor features
        descriptor_features = []
        for descriptor in self.descriptor_list:
            feature = descriptor.transform_pmg(pmg_obj, site_index=target_site_indices)
            descriptor_features.append(feature)
        descriptor_features = np.concatenate(descriptor_features, axis=1)

        seq = 0
        for site_idx, df in zip(target_site_indices, descriptor_features):
            # descriptor features
            df = torch.tensor(df, dtype=torch.float32)

            # Target-site atomic number
            element = torch.tensor(pmg_obj.atomic_numbers[site_idx], dtype=torch.int64)

            # Spectrum
            spectrum = pmg_obj.site_properties["spectrum"][site_idx]
            energies = torch.tensor(spectrum["energies"], dtype=torch.float32)
            intensities = torch.tensor(spectrum["intensities"], dtype=torch.float32)

            # Assign x (input) and y (target) based on direction
            if self._inverse:
                x = intensities
                y = df
            else:
                x = df
                y = intensities

            # Create Data object
            data = DescriptorData(
                x=x,
                y=y,
                energies=energies,
                sample_id=pmg_obj.properties["sample_id"],
                element=element,
                target_site_index=torch.tensor(site_idx, dtype=torch.int64),
            )

            # Save processed data
            data.save(save_path_fn(seq))
            seq += 1

        return seq

    def collate_fn(self, batch: list[DescriptorData]) -> DescriptorData:
        """Collate descriptor samples into a batch.

        Args:
            batch: Descriptor samples loaded by ``__getitem__``.

        Returns:
            Batched descriptor data with stacked tensor fields.
        """

        def _stack(tensors: list[torch.Tensor | None]) -> torch.Tensor:
            """Stack tensors along a new leading batch dimension."""
            return torch.stack([cast(torch.Tensor, tensor) for tensor in tensors], dim=0)

        return DescriptorData(
            x=_stack([b.x for b in batch]),
            y=_stack([b.y for b in batch]),
            energies=_stack([b.energies for b in batch]),
            sample_id=[b.sample_id for b in batch],
            element=_stack([b.element for b in batch]),
            target_site_index=_stack([b.target_site_index for b in batch]),
        )

    def _load_item(self, path: str) -> DescriptorData:
        """Load one processed descriptor sample.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Loaded descriptor data object.
        """
        return DescriptorData.load(path)

    @property
    def signature(self) -> Config:
        """Dataset configuration signature.

        Returns:
            Configuration values that identify this descriptor dataset.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "descriptors": self.descriptor_configs,
            }
        )
        return signature
