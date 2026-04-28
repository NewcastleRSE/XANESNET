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

"""Multiprocessing geometry graph dataset registration."""

from xanesnet.datasets._mp import MpDatasetMixin
from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ...registry import DatasetRegistry
from ..geometrygraph import GeometryGraphDataset


@DatasetRegistry.register("geometrygraph_mp")
class GeometryGraphDatasetMp(MpDatasetMixin, GeometryGraphDataset):
    """Multiprocessing variant of :class:`GeometryGraphDataset`.

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw datasource used during preparation.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        cutoff: Graph cutoff in **Angstrom**.
        max_num_neighbors: Per-source neighbor cap.
        compute_angles: Whether to precompute triplet angles.
        graph_method: Graph construction method.
        min_facet_area: Optional Voronoi facet-area threshold.
        cov_radii_scale: Covalent-radii scale for graph construction.
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
        # params
        cutoff: float,
        max_num_neighbors: int,
        compute_angles: bool,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
        num_workers: int | None,
    ) -> None:
        """Initialize a multiprocessing geometry graph dataset."""
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
            compute_angles=compute_angles,
            graph_method=graph_method,
            min_facet_area=min_facet_area,
            cov_radii_scale=cov_radii_scale,
        )
        self.num_workers = num_workers
