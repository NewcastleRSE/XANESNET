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

"""Z-score standardization spectra encoding for XANESNET."""

import torch

from xanesnet.serialization.auto_config.registries import EncodingAutoResolver
from xanesnet.serialization.auto_config.statistics import SpectralStatisticsCollector
from xanesnet.serialization.config import Config, ConfigRaw
from xanesnet.utils.exceptions import ConfigError

from .affine import AffineEncoding, build_parameter_rows
from .registry import SpectraEncodingRegistry


@SpectraEncodingRegistry.register("z_score")
class ZScoreEncoding(AffineEncoding):
    """Z-score standardization encoding.

    Standardizes spectra during encoding using ``(x - mean) / std`` and inverts
    the transform during decoding using ``x * std + mean``. The ``mean`` and
    ``std`` are typically the mean and population standard deviation of the
    training spectra and can be provided explicitly or resolved automatically
    from the training data via the ``"auto"`` token.

    Two orthogonal switches control how the statistics are shaped:

    * ``per_point`` selects whether the statistics vary across the spectrum
      (a length-``N`` vector) or are a single global scalar shared across all
      points.
    * ``per_element`` selects whether a separate set of statistics is used for
    each target-site element (chosen per sample from its atomic number) or a
      single set is shared across all elements.

    When ``per_element`` is false, ``mean`` and ``std`` are flat lists. When it
    is true, they are lists of rows, one row per entry of ``elements``.

    Zero or near-zero standard deviations (below ``1e-12``) are silently
    replaced with ``1.0``, which is equivalent to skipping scaling at those
    points (only centering is applied). This handles flat spectral regions
    where the per-point standard deviation vanishes.

    Args:
        encoding_type: Identifier string for this encoding type.
        mean: Means used to center the spectra. A flat list when
            ``per_element`` is false, or one row per element (aligned with
            ``elements``) when true.
        std: Standard deviations used to scale the spectra, shaped like
            ``mean``.
        per_point: Whether standardization is applied per spectrum point
            (``True``) or globally with a single shared statistic (``False``).
        per_element: Whether statistics are selected per target-site element
            (``True``) or shared across all elements (``False``).
        elements: Atomic numbers aligned row-wise with ``mean`` and ``std`` when
            ``per_element`` is true; ignored otherwise.

    Raises:
        ConfigError: If ``mean`` or ``std`` is empty, or the two are
            misaligned.
    """

    def __init__(
        self,
        encoding_type: str,
        mean: list[float] | list[list[float]],
        std: list[float] | list[list[float]],
        per_point: bool,
        per_element: bool,
        elements: list[int] | None,
    ) -> None:
        """Initialize ``ZScoreEncoding``."""
        if per_element:
            shift = build_parameter_rows(encoding_type, "mean", mean)  # type: ignore[arg-type]
            scale = build_parameter_rows(encoding_type, "std", std)  # type: ignore[arg-type]
            if shift.shape != scale.shape:
                raise ConfigError(
                    f"{encoding_type} mean and std shapes mismatch: {tuple(shift.shape)} vs {tuple(scale.shape)}."
                )
        else:
            if not mean or not std:
                raise ConfigError(f"{encoding_type} requires non-empty mean and std.")
            if len(mean) != len(std):
                raise ConfigError(f"{encoding_type} mean and std length mismatch: {len(mean)} vs {len(std)}.")
            shift = torch.tensor(mean, dtype=torch.float32)
            scale = torch.tensor(std, dtype=torch.float32)

        # Replace near-zero std values with 1.0 (identity scaling) so that
        # flat spectral regions where std \approx 0 do not
        # cause division-by-zero during encoding.
        scale = torch.where(
            scale < 1e-12,
            torch.ones_like(scale),
            scale,
        )

        self.per_point = per_point
        super().__init__(encoding_type, shift, scale, per_element, elements)

    @property
    def signature(self) -> list[Config]:
        """Return the z-score-encoding signature.

        Returns:
            Configuration values needed to recreate this encoding.
        """
        sig = super().signature[0]
        values: dict[str, object] = {
            "mean": self.shift.tolist(),
            "std": self.scale.tolist(),
            "per_point": self.per_point,
            "per_element": self.per_element,
        }
        if self.per_element:
            values["elements"] = list(self.elements or [])
        sig.update_with_dict(values)
        return [sig]


@EncodingAutoResolver.register("z_score")
def resolve_zscore_encoding(item: ConfigRaw, statistics: SpectralStatisticsCollector) -> ConfigRaw:
    """Resolve z-score statistics from the training spectra.

    Args:
        item: Raw z-score encoding configuration dictionary.
        statistics: Streaming statistics of the training spectra
            (:class:`~xanesnet.serialization.auto_config.statistics.SpectralStatisticsCollector`).

    Returns:
        Mapping with ``mean`` and population ``std``.  When ``per_element`` is
        false these are per-point lists (or single-element lists when
        ``per_point`` is false); when true they are one row per element
        together with the resolved ``elements``.
    """
    per_point = item.get("per_point", True)
    if item.get("per_element", False):
        requested = item.get("elements")
        elements = [int(z) for z in requested] if isinstance(requested, list) else statistics.sorted_elements()
        mean_rows: list[list[float]] = []
        std_rows: list[list[float]] = []
        for z in elements:
            stats = statistics.element_stats(z)
            if per_point:
                mean_rows.append(stats.mean.tolist())
                std_rows.append(stats.std.tolist())
            else:
                mean_rows.append([stats.global_mean])
                std_rows.append([stats.global_std])
        return {"elements": list(elements), "mean": mean_rows, "std": std_rows}

    stats = statistics.overall
    if per_point:
        return {"mean": stats.mean.tolist(), "std": stats.std.tolist()}
    return {"mean": [stats.global_mean], "std": [stats.global_std]}
