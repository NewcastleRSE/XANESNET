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

"""Multiprocessing descriptor-based multi-head dataset registration."""

from xanesnet.datasets._mp import MpDatasetMixin
from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ...registry import DatasetRegistry
from ..descriptor_multihead import DescriptorMultiheadDataset


@DatasetRegistry.register("descriptor_multihead_mp")
@DatasetRegistry.register("descriptor_multihead_inverse_mp")
class DescriptorMultiheadDatasetMp(MpDatasetMixin, DescriptorMultiheadDataset):
    """Multiprocessing variant of :class:`DescriptorMultiheadDataset`.

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw datasource used during preparation.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        descriptors: Descriptor configuration objects.
        num_workers: Requested worker process count.
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
        descriptors: list[Config],
        num_workers: int | None,
    ) -> None:
        """Initialize a multiprocessing descriptor multi-head dataset."""
        super().__init__(
            dataset_type=dataset_type,
            datasource=datasource,
            root=root,
            preload=preload,
            skip_prepare=skip_prepare,
            split_ratios=split_ratios,
            split_indexfile=split_indexfile,
            descriptors=descriptors,
        )
        self.num_workers = num_workers
