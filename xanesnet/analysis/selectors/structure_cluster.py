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

"""Selector that partitions structures into clusters and selects one cluster."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from xanesnet.serialization.config import Config
from xanesnet.serialization.prediction_readers import PredictionReader
from xanesnet.utils.exceptions import ConfigError

from ..descriptor_cache import descriptor_matrix
from .base import Selector
from .registry import SelectorRegistry

# Label used for members of clusters that stay below ``min_cluster_size``.
OTHERS_LABEL = -1


@dataclass(frozen=True)
class _Clustering:
    """Cached clustering result for one reader and parameter set."""

    indices_by_cluster: dict[int, list[int]]
    cluster_ids: tuple[int, ...]
    has_others: bool


# Cache shared across selector instances so the clustering is computed once per
# prediction reader instead of once per expanded cluster selector.
_CLUSTER_CACHE: dict[tuple[Any, ...], _Clustering] = {}


@SelectorRegistry.register("structure_cluster")
class StructureClusterSelector(Selector):
    """Select the samples of one structure cluster.

    Structures are embedded with a descriptor and partitioned by agglomerative
    hierarchical clustering (ward linkage). One instance keeps the samples of a
    single ``cluster_id``; ``-1`` selects the ``others`` bucket holding samples
    from clusters smaller than ``min_cluster_size``. With ``cluster_id=None``
    the selector expands into one instance per cluster. Clustering is computed
    once per reader and reused across expanded instances.

    Requires:
        Matched raw structures: provided by a structure-matched prediction reader.

    Args:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader to select samples from.
        descriptor: Descriptor configuration object used to embed each structure.
        cluster_distance: Linkage-distance cut for cluster formation. ``None``
            uses ``distance_fraction`` of the maximum linkage height.
        distance_fraction: Fraction of the maximum linkage height used as the
            cut when ``cluster_distance`` is ``None``.
        min_cluster_size: Clusters with fewer members are folded into the
            ``others`` bucket.
        cluster_id: Cluster to select. ``None`` expands into one selector per
            cluster plus the ``others`` bucket (``-1``); an integer selects only
            that cluster (skipped if it does not exist).
    """

    def __init__(
        self,
        selector_type: str,
        data_source: PredictionReader,
        descriptor: Config,
        cluster_distance: float | None,
        distance_fraction: float,
        min_cluster_size: int,
        cluster_id: int | None,
    ) -> None:
        """Initialize a structure cluster selector.

        Raises:
            ConfigError: If ``data_source`` does not attach matched raw
                structures or holds fewer than two structures.
        """
        super().__init__(selector_type, data_source)

        if not data_source.provides_structures():
            raise ConfigError("StructureClusterSelector requires predictions matched to raw structures.")

        clustering = _get_clustering(
            data_source,
            descriptor,
            cluster_distance,
            distance_fraction,
            min_cluster_size,
        )
        self.cluster_id = cluster_id
        self.cluster_ids = list(clustering.cluster_ids)
        self.has_others = clustering.has_others
        self._indices = [] if cluster_id is None else list(clustering.indices_by_cluster.get(cluster_id, []))
        self.descriptor_config = descriptor
        self.cluster_distance = cluster_distance
        self.distance_fraction = distance_fraction
        self.min_cluster_size = min_cluster_size

    def __iter__(self):
        """Yield the samples of the configured cluster.

        Returns:
            Iterator over selected prediction samples.
        """
        for index in self._indices:
            yield self.data_source[index]

    def __len__(self) -> int:
        """Return the number of samples in the configured cluster.

        Returns:
            Number of selected prediction samples.
        """
        return len(self._indices)

    def expand_selectors(self) -> list[Selector]:
        """Return the selectors this selector expands into.

        With ``cluster_id=None`` this returns one selector per cluster plus the
        ``others`` bucket (``-1``) when present. With an explicit ``cluster_id``
        it returns only that cluster, or an empty list when the cluster does not
        exist.

        Returns:
            One selector per requested cluster, or an empty list when an
            explicit cluster does not exist.
        """
        if self.cluster_id is not None:
            if self.cluster_id == OTHERS_LABEL and not self.has_others:
                logging.warning("No 'others' bucket exists; skipping cluster_id=-1.")
                return []
            if self.cluster_id != OTHERS_LABEL and self.cluster_id not in self.cluster_ids:
                logging.warning(
                    "Cluster %s does not exist (available: %s); skipping selector.",
                    self.cluster_id,
                    self.cluster_ids,
                )
                return []
            return [self]

        cluster_ids = list(self.cluster_ids)
        if self.has_others:
            cluster_ids.append(OTHERS_LABEL)
        return [
            type(self)(
                selector_type=self.selector_type,
                data_source=self.data_source,
                descriptor=self.descriptor_config,
                cluster_distance=self.cluster_distance,
                distance_fraction=self.distance_fraction,
                min_cluster_size=self.min_cluster_size,
                cluster_id=cluster_id,
            )
            for cluster_id in cluster_ids
        ]

    @property
    def signature(self) -> Config:
        """Return the selector signature.

        Returns:
            Configuration values needed to recreate this selector.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "descriptor": self.descriptor_config.as_dict(),
                "cluster_distance": self.cluster_distance,
                "distance_fraction": self.distance_fraction,
                "min_cluster_size": self.min_cluster_size,
                "cluster_id": self.cluster_id,
            }
        )
        return signature

    def __str__(self) -> str:
        """Return the short display label of this selector."""
        return f"C{self.cluster_id}"


def _get_clustering(
    data_source: PredictionReader,
    descriptor: Config,
    cluster_distance: float | None,
    distance_fraction: float,
    min_cluster_size: int,
) -> _Clustering:
    """Return the cached clustering for one reader and descriptor config.

    Args:
        data_source: Structure-matched prediction reader to cluster.
        descriptor: Descriptor configuration used to embed each structure.
        cluster_distance: Optional linkage-distance cut.
        distance_fraction: Fraction of the maximum linkage height used when
            ``cluster_distance`` is ``None``.
        min_cluster_size: Minimum size of a retained cluster.

    Returns:
        Cached or newly computed clustering result.
    """
    key = (
        id(data_source),
        repr(sorted(descriptor.as_dict().items())),
        cluster_distance,
        distance_fraction,
        min_cluster_size,
    )
    if key not in _CLUSTER_CACHE:
        _CLUSTER_CACHE[key] = _compute_clustering(
            data_source, descriptor, cluster_distance, distance_fraction, min_cluster_size
        )
    return _CLUSTER_CACHE[key]


def _compute_clustering(
    data_source: PredictionReader,
    descriptor: Config,
    cluster_distance: float | None,
    distance_fraction: float,
    min_cluster_size: int,
) -> _Clustering:
    """Embed, standardize, and cluster every structure of one reader.

    Args:
        data_source: Structure-matched prediction reader to cluster.
        descriptor: Descriptor configuration used to embed each structure.
        cluster_distance: Optional linkage-distance cut.
        distance_fraction: Fraction of the maximum linkage height used when
            ``cluster_distance`` is ``None``.
        min_cluster_size: Minimum size of a retained cluster.

    Returns:
        Cluster membership and the IDs available for selector expansion.

    Raises:
        ConfigError: If fewer than two structures are available.
    """
    if len(data_source) < 2:
        raise ConfigError("StructureClusterSelector requires at least two samples with structures.")

    features = descriptor_matrix(descriptor, data_source)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = (features - features.mean(axis=0)) / features.std(axis=0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    z = linkage(scaled, method="ward")
    threshold = cluster_distance if cluster_distance is not None else distance_fraction * float(z[-1, 2])
    raw = fcluster(z, t=threshold, criterion="distance")

    counts = np.bincount(raw)
    small = {i for i, count in enumerate(counts) if count > 0 and count < min_cluster_size}
    mapping: dict[int, int] = {}
    for raw_id in raw:
        key = int(raw_id) - 1
        if int(raw_id) not in small and key not in mapping:
            mapping[key] = len(mapping)
    labels = np.array(
        [OTHERS_LABEL if int(raw_id) in small else mapping[int(raw_id) - 1] for raw_id in raw],
        dtype=int,
    )

    indices_by_cluster: dict[int, list[int]] = {cid: [] for cid in range(len(mapping))}
    indices_by_cluster[OTHERS_LABEL] = []
    for idx, label in enumerate(labels):
        indices_by_cluster[int(label)].append(idx)

    has_others = bool(indices_by_cluster[OTHERS_LABEL])
    if not has_others:
        indices_by_cluster.pop(OTHERS_LABEL)

    cluster_ids = tuple(cid for cid in sorted(indices_by_cluster) if cid != OTHERS_LABEL)
    return _Clustering(indices_by_cluster=indices_by_cluster, cluster_ids=cluster_ids, has_others=has_others)
