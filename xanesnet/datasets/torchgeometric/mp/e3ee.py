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

"""Multiprocessing E3EE dataset registration."""

from xanesnet.datasets._mp import MpDatasetMixin
from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ...registry import DatasetRegistry
from ..e3ee import E3EEDataset


@DatasetRegistry.register("e3ee_mp")
class E3EEDatasetMp(MpDatasetMixin, E3EEDataset):
    """Multiprocessing variant of :class:`E3EEDataset`. See the parent for semantics."""

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
        use_path_branch: bool,
        max_paths_per_structure: int,
        graph_method: str,
        min_facet_area: float | str | None,
        cov_radii_scale: float,
        att_cutoff: float,
        att_max_num_neighbors: int,
        att_graph_method: str,
        att_min_facet_area: float | str | None,
        att_cov_radii_scale: float,
        num_workers: int | None,
    ) -> None:
        """Initialize a multiprocessing E3EE dataset.

        Args:
            dataset_type: Registered dataset type name.
            datasource: Raw datasource used during preparation.
            root: Directory that stores processed ``.pth`` files.
            preload: Whether to preload processed samples.
            skip_prepare: Whether to reuse existing processed files.
            split_ratios: Optional split ratios.
            split_indexfile: Optional path to split indices.
            cutoff: Main graph cutoff in **Angstrom**.
            max_num_neighbors: Main graph per-source neighbor cap.
            use_path_branch: Whether to precompute absorber-centered paths.
            max_paths_per_structure: Maximum absorber paths saved per structure.
            graph_method: Main graph construction method.
            min_facet_area: Optional Voronoi facet-area threshold.
            cov_radii_scale: Covalent-radii scale for graph construction.
            att_cutoff: Attention graph cutoff in **Angstrom**.
            att_max_num_neighbors: Attention graph per-source neighbor cap.
            att_graph_method: Attention graph construction method.
            att_min_facet_area: Optional attention Voronoi facet-area threshold.
            att_cov_radii_scale: Attention graph covalent-radii scale.
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
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            use_path_branch=use_path_branch,
            max_paths_per_structure=max_paths_per_structure,
            graph_method=graph_method,
            min_facet_area=min_facet_area,
            cov_radii_scale=cov_radii_scale,
            att_cutoff=att_cutoff,
            att_max_num_neighbors=att_max_num_neighbors,
            att_graph_method=att_graph_method,
            att_min_facet_area=att_min_facet_area,
            att_cov_radii_scale=att_cov_radii_scale,
        )
        self.num_workers = num_workers
