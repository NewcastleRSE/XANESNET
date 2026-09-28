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

"""Streaming spectral statistics for automatic encoding configuration.

Provides :class:`SpectralStatistics` (per-point running accumulator) and
:class:`SpectralStatisticsCollector` (overall + per-element wrapper) that
accumulate training spectral statistics in a single pass without materialising
the full spectral matrix, plus a function :func:`collect_spectral_statistics`
that populates a collector from a prepared :class:`~xanesnet.datasets.Dataset`.
"""

import torch
from tqdm import tqdm

from xanesnet.batchprocessors import BatchProcessor, InverseBatchProcessor
from xanesnet.datasets import Dataset
from xanesnet.utils.exceptions import ConfigError

# Fixed data-loader settings for the statistics pass.
STATISTICS_BATCH_SIZE: int = 32
STATISTICS_NUM_WORKERS: int = 4


class SpectralStatistics:
    """Streaming per-point statistics over training spectra.

    Folds batches of spectra ``(B, N)`` into fixed-size per-point
    accumulators (count, sum, sum of squares, running minimum, and running
    maximum), so dataset-dependent encoding parameters can be derived in a
    single pass without materializing the full spectral matrix.  Sums are
    accumulated in double precision for numerical stability.
    """

    def __init__(self) -> None:
        """Initialize an empty accumulator."""
        self._count = 0
        self._sum = torch.empty(0, dtype=torch.float64)
        self._sum_sq = torch.empty(0, dtype=torch.float64)
        self._minimum = torch.empty(0, dtype=torch.float64)
        self._maximum = torch.empty(0, dtype=torch.float64)

    def update(self, spectra: torch.Tensor) -> None:
        """Fold one batch of training spectra into the accumulators.

        Args:
            spectra: Training spectra ``(B, N)``.
        """
        values = spectra.to(dtype=torch.float64)
        batch_sum = values.sum(dim=0)
        batch_sum_sq = values.square().sum(dim=0)
        batch_min = values.amin(dim=0)
        batch_max = values.amax(dim=0)

        if self._count == 0:
            self._sum = batch_sum
            self._sum_sq = batch_sum_sq
            self._minimum = batch_min
            self._maximum = batch_max
        else:
            self._sum += batch_sum
            self._sum_sq += batch_sum_sq
            self._minimum = torch.minimum(self._minimum, batch_min)
            self._maximum = torch.maximum(self._maximum, batch_max)

        self._count += values.shape[0]

    @property
    def num_points(self) -> int:
        """Number of points ``N`` in the accumulated spectra."""
        return int(self._sum.shape[-1])

    @property
    def mean(self) -> torch.Tensor:
        """Per-point mean of the accumulated spectra."""
        return self._sum / self._count

    @property
    def std(self) -> torch.Tensor:
        """Per-point population standard deviation of the accumulated spectra."""
        variance = self._sum_sq / self._count - self.mean.square()
        return variance.clamp_min(0.0).sqrt()

    @property
    def minimum(self) -> torch.Tensor:
        """Per-point minimum of the accumulated spectra."""
        return self._minimum

    @property
    def maximum(self) -> torch.Tensor:
        """Per-point maximum of the accumulated spectra."""
        return self._maximum

    @property
    def global_mean(self) -> float:
        """Scalar mean over every point of the accumulated spectra."""
        return float(self._sum.sum() / (self._count * self.num_points))

    @property
    def global_std(self) -> float:
        """Scalar population standard deviation over every point."""
        total = self._count * self.num_points
        mean = self._sum.sum() / total
        variance = self._sum_sq.sum() / total - mean.square()
        return float(variance.clamp_min(0.0).sqrt())

    @property
    def global_minimum(self) -> float:
        """Scalar minimum over every point of the accumulated spectra."""
        return float(self._minimum.min())

    @property
    def global_maximum(self) -> float:
        """Scalar maximum over every point of the accumulated spectra."""
        return float(self._maximum.max())


class SpectralStatisticsCollector:
    """Overall and per-element streaming statistics over training spectra.

    Maintains one :class:`SpectralStatistics` accumulator across all spectra
    and one accumulator per target-site element. Spectral rows are routed to
    their element bucket using the atomic numbers supplied alongside each
    batch; when no element information is available only the overall
    accumulator is updated.
    """

    def __init__(self) -> None:
        """Initialize empty overall and per-element accumulators."""
        self.overall = SpectralStatistics()
        self.per_element: dict[int, SpectralStatistics] = {}

    def update(self, spectra: torch.Tensor, elements: torch.Tensor | None = None) -> None:
        """Fold one batch of training spectra into the accumulators.

        Args:
            spectra: Training spectra ``(B, N)``.
            elements: Per-row target-site atomic numbers ``(B,)``, or ``None``
                when the batch carries no element information.
        """
        self.overall.update(spectra)
        if elements is None:
            return

        atomic_numbers = elements.to(dtype=torch.int64).reshape(-1)
        for z in torch.unique(atomic_numbers).tolist():
            mask = atomic_numbers == z
            self.per_element.setdefault(int(z), SpectralStatistics()).update(spectra[mask])

    def sorted_elements(self) -> list[int]:
        """Return the observed atomic numbers in ascending order.

        Returns:
            Sorted list of atomic numbers seen during accumulation.
        """
        return sorted(self.per_element)

    def element_stats(self, element: int) -> SpectralStatistics:
        """Return per-element statistics, raising if the element is absent.

        Args:
            element: Atomic number to look up.

        Returns:
            Statistics accumulated for the requested element.

        Raises:
            ConfigError: If no training spectrum carries the requested
                element.
        """
        stats = self.per_element.get(int(element))
        if stats is None:
            raise ConfigError(f"Element {element} not found in training spectra.")
        return stats


def collect_spectral_statistics(dataset: Dataset, batchprocessor: BatchProcessor) -> SpectralStatisticsCollector:
    """Accumulate training spectral statistics in a single batched pass.

    Iterates the training subset (or the whole dataset when no split is
    configured) with the dataset's own data loader, so samples are loaded in
    parallel worker processes and collated in batches instead of one at a
    time. The batch size and worker count are the fixed module constants
    ``STATISTICS_BATCH_SIZE`` and ``STATISTICS_NUM_WORKERS``. Each batch is
    folded into a running :class:`SpectralStatisticsCollector`, which
    maintains both overall and per-target-site-element statistics. For
    forward batch processors the spectrum is the model target; for inverse
    batch processors it is the spectral input (under
    :attr:`~xanesnet.batchprocessors.InverseBatchProcessor.spectra_input_key`).
    Only fixed-size accumulators are held in memory, so the full spectral
    matrix is never materialized.

    Args:
        dataset: Prepared training dataset.
        batchprocessor: Batch processor for the dataset/model pair.

    Returns:
        Streaming statistics over the training spectra.
    """
    subset = dataset.train_subset if dataset.train_subset is not None else dataset

    dataloader_cls = dataset.get_dataloader()
    dataloader = dataloader_cls(
        subset,
        batch_size=STATISTICS_BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
        num_workers=STATISTICS_NUM_WORKERS,
        pin_memory=False,
        persistent_workers=STATISTICS_NUM_WORKERS > 0,
        prefetch_factor=2 if STATISTICS_NUM_WORKERS > 0 else None,
    )

    collector = SpectralStatisticsCollector()
    for batch in tqdm(dataloader, desc="Collecting spectral statistics"):
        if isinstance(batchprocessor, InverseBatchProcessor):
            spectra = batchprocessor.input_preparation(batch)[batchprocessor.spectra_input_key]
        else:
            spectra = batchprocessor.target_preparation(batch)
        elements = batchprocessor.element_preparation(batch)
        collector.update(spectra, elements)
    return collector
