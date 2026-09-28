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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Multiprocessing GemNet and GemNet-OC dataset registration."""

from xanesnet.datasets._mp import MpDatasetMixin
from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from ...registry import DatasetRegistry
from ..gemnet import GemNetDataset


@DatasetRegistry.register("gemnet_mp")
@DatasetRegistry.register("gemnet_oc_mp")
class GemNetDatasetMp(MpDatasetMixin, GemNetDataset):
    """Multiprocessing variant of :class:`GemNetDataset` (covers GemNet and GemNet-OC).

    Args:
        dataset_type: Registered dataset type name.
        datasource: Raw datasource used during preparation.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        graph_builder: Main graph builder configuration.
        quadruplets: Whether to compute quadruplet indices.
        int_graph_builder: Interaction graph builder configuration. When
            ``None`` the main ``graph_builder`` is reused.
        oc_mode: Whether to precompute GemNet-OC auxiliary graphs and mixed triplets.
        oc_aeaint_graph_builder: Atom-edge-atom graph builder configuration.
        oc_aint_graph_builder: Atom-atom graph builder configuration.
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
        graph_builder: Config,
        quadruplets: bool,
        int_graph_builder: Config | None = None,
        oc_mode: bool = False,
        oc_aeaint_graph_builder: Config | None = None,
        oc_aint_graph_builder: Config | None = None,
        num_workers: int | None = None,
    ) -> None:
        """Initialize a multiprocessing GemNet dataset."""
        super().__init__(
            dataset_type=dataset_type,
            datasource=datasource,
            root=root,
            preload=preload,
            skip_prepare=skip_prepare,
            split_ratios=split_ratios,
            split_indexfile=split_indexfile,
            graph_builder=graph_builder,
            quadruplets=quadruplets,
            int_graph_builder=int_graph_builder,
            oc_mode=oc_mode,
            oc_aeaint_graph_builder=oc_aeaint_graph_builder,
            oc_aint_graph_builder=oc_aint_graph_builder,
        )
        self.num_workers = num_workers
