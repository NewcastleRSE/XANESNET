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

"""Min-max normalization spectra encoding for XANESNET."""

import torch

from xanesnet.serialization.auto_config.registries import EncodingAutoResolver
from xanesnet.serialization.auto_config.statistics import SpectralStatisticsCollector
from xanesnet.serialization.config import Config, ConfigRaw
from xanesnet.utils.exceptions import ConfigError

from .affine import AffineEncoding, build_parameter_rows
from .registry import SpectraEncodingRegistry


@SpectraEncodingRegistry.register("min_max")
class MinMaxEncoding(AffineEncoding):
    """Min-max normalization encoding.

    Normalizes spectra during encoding into the unit interval using
    ``(x - minimum) / (maximum - minimum)`` and inverts the transform during
    decoding using ``x * (maximum - minimum) + minimum``. The ``minimum`` and
    ``maximum`` are typically the minimum and maximum of the training spectra
    and can be provided explicitly or resolved automatically from the training
    data via the ``"auto"`` token.

    Two orthogonal switches control how the bounds are shaped. ``per_point``
    selects whether the bounds vary across the spectrum (a length-``N`` vector)
    or are a single global scalar shared across all points. ``per_element``
    selects whether a separate set of bounds is used for each target-site
    element (chosen per sample from its atomic number) or a single set is shared
    across all elements.

    When ``per_element`` is false, ``minimum`` and ``maximum`` are flat lists.
    When it is true, they are lists of rows, one row per entry of ``elements``.

    Zero or near-zero spans (``maximum - minimum`` below ``1e-12``) are
    silently replaced with ``1.0``, which is equivalent to skipping scaling at
    those points. This handles flat spectral regions where
    all training spectra share the same value.

    Args:
        encoding_type: Identifier string for this encoding type.
        minimum: Minima used during normalization. A flat list when
            ``per_element`` is false, or one row per element (aligned with
            ``elements``) when true.
        maximum: Maxima used during normalization, shaped like ``minimum``.
        per_point: Whether normalization is applied per spectrum point
            (``True``) or globally with a single shared statistic (``False``).
        per_element: Whether bounds are selected per target-site element
            (``True``) or shared across all elements (``False``).
        elements: Atomic numbers aligned row-wise with ``minimum`` and
            ``maximum`` when ``per_element`` is true; ignored otherwise.

    Raises:
        ConfigError: If ``minimum`` or ``maximum`` is empty, or the two are
            misaligned.
    """

    def __init__(
        self,
        encoding_type: str,
        minimum: list[float] | list[list[float]],
        maximum: list[float] | list[list[float]],
        per_point: bool,
        per_element: bool,
        elements: list[int] | None,
    ) -> None:
        """Initialize ``MinMaxEncoding``."""
        if per_element:
            minimum_tensor = build_parameter_rows(encoding_type, "minimum", minimum)  # type: ignore[arg-type]
            maximum_tensor = build_parameter_rows(encoding_type, "maximum", maximum)  # type: ignore[arg-type]
            if minimum_tensor.shape != maximum_tensor.shape:
                raise ConfigError(
                    f"{encoding_type} minimum and maximum shapes mismatch: "
                    f"{tuple(minimum_tensor.shape)} vs {tuple(maximum_tensor.shape)}."
                )
        else:
            if not minimum or not maximum:
                raise ConfigError(f"{encoding_type} requires non-empty minimum and maximum.")
            if len(minimum) != len(maximum):
                raise ConfigError(
                    f"{encoding_type} minimum and maximum length mismatch: {len(minimum)} vs {len(maximum)}."
                )
            minimum_tensor = torch.tensor(minimum, dtype=torch.float32)
            maximum_tensor = torch.tensor(maximum, dtype=torch.float32)

        span = maximum_tensor - minimum_tensor
        # Replace near-zero spans with 1.0 (identity scaling) so that flat
        # spectral regions where max \approx min do not cause
        # division-by-zero during encoding.
        span = torch.where(
            span < 1e-12,
            torch.ones_like(span),
            span,
        )

        self.per_point = per_point
        super().__init__(encoding_type, minimum_tensor, span, per_element, elements)

    @property
    def signature(self) -> list[Config]:
        """Return the min-max-encoding signature.

        The stored ``shift`` holds the minimum and ``scale`` holds the span, so
        the maximum is recovered as ``shift + scale``.

        Returns:
            Configuration values needed to recreate this encoding.
        """
        sig = super().signature[0]
        values: dict[str, object] = {
            "minimum": self.shift.tolist(),
            "maximum": (self.shift + self.scale).tolist(),
            "per_point": self.per_point,
            "per_element": self.per_element,
        }
        if self.per_element:
            values["elements"] = list(self.elements or [])
        sig.update_with_dict(values)
        return [sig]


@EncodingAutoResolver.register("min_max")
def resolve_minmax_encoding(item: ConfigRaw, statistics: SpectralStatisticsCollector) -> ConfigRaw:
    """Resolve minima and maxima from the training spectra.

    Args:
        item: Raw min-max encoding configuration dictionary.
        statistics: Streaming statistics of the training spectra
            (:class:`~xanesnet.serialization.auto_config.statistics.SpectralStatisticsCollector`).

    Returns:
        Mapping with ``minimum`` and ``maximum``.  When ``per_element`` is
        false these are per-point lists (or single-element lists when
        ``per_point`` is false); when true they are one row per element
        together with the resolved ``elements``.
    """
    per_point = item.get("per_point", True)
    if item.get("per_element", False):
        requested = item.get("elements")
        elements = [int(z) for z in requested] if isinstance(requested, list) else statistics.sorted_elements()
        minimum_rows: list[list[float]] = []
        maximum_rows: list[list[float]] = []
        for z in elements:
            stats = statistics.element_stats(z)
            if per_point:
                minimum_rows.append(stats.minimum.tolist())
                maximum_rows.append(stats.maximum.tolist())
            else:
                minimum_rows.append([stats.global_minimum])
                maximum_rows.append([stats.global_maximum])
        return {"elements": list(elements), "minimum": minimum_rows, "maximum": maximum_rows}

    stats = statistics.overall
    if per_point:
        return {"minimum": stats.minimum.tolist(), "maximum": stats.maximum.tolist()}
    return {"minimum": [stats.global_minimum], "maximum": [stats.global_maximum]}
