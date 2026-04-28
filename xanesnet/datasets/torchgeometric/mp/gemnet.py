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

from xanesnet.datasets._mp import MpDatasetMixin
from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ...registry import DatasetRegistry
from ..gemnet import GemNetDataset


@DatasetRegistry.register("gemnet_mp")
@DatasetRegistry.register("gemnet_oc_mp")
class GemNetDatasetMp(MpDatasetMixin, GemNetDataset):
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
