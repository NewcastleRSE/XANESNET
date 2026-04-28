# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either Version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

"""Multiprocessing environment-embedding dataset registration."""

from xanesnet.datasets._mp import MpDatasetMixin
from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ...registry import DatasetRegistry
from ..envembed import EnvEmbedDataset


@DatasetRegistry.register("envembed_mp")
class EnvEmbedDatasetMp(MpDatasetMixin, EnvEmbedDataset):
    """Multiprocessing variant of :class:`EnvEmbedDataset`."""

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
        widths_eV: list[float],
        basis_stride: int,
        basis_path: str | None,
        env_radius: float | None,
        descriptors: list[Config],
        num_workers: int | None,
    ) -> None:
        """Initialize a multiprocessing environment-embedding dataset.

        Args:
            dataset_type: Registered dataset type name.
            datasource: Raw datasource used during preparation.
            root: Directory that stores processed ``.pth`` files.
            preload: Whether to preload processed samples.
            skip_prepare: Whether to reuse existing processed files.
            split_ratios: Optional split ratios.
            split_indexfile: Optional path to split indices.
            widths_eV: Gaussian basis widths in **eV**.
            basis_stride: Energy-grid stride used when creating a Gaussian basis.
            basis_path: Optional serialized spectral basis path.
            env_radius: Periodic-neighbor cutoff in **Angstrom**.
            descriptors: Descriptor configuration objects.
            num_workers: Requested worker process count.
        """
        super().__init__(
            dataset_type=dataset_type,
            datasource=datasource,
            root=root,
            preload=preload,
            skip_prepare=skip_prepare,
            split_ratios=split_ratios,
            split_indexfile=split_indexfile,
            widths_eV=widths_eV,
            basis_stride=basis_stride,
            basis_path=basis_path,
            env_radius=env_radius,
            descriptors=descriptors,
        )
        self.num_workers = num_workers
