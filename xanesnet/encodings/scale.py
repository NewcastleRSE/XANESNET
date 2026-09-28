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

"""Factor-scaling spectra encoding for XANESNET."""

import torch

from xanesnet.serialization.auto_config.registries import EncodingAutoResolver
from xanesnet.serialization.auto_config.statistics import SpectralStatisticsCollector
from xanesnet.serialization.config import Config, ConfigRaw
from xanesnet.utils.exceptions import ConfigError

from .affine import AffineEncoding, build_parameter_rows
from .registry import SpectraEncodingRegistry


@SpectraEncodingRegistry.register("scale")
class ScaleEncoding(AffineEncoding):
    """Factor-scaling encoding.

    Divides spectra by a ``factor`` during encoding (``x / factor``) and
    multiplies by the same factor during decoding (``x * factor``), without any
    centering. Typically the factor is the standard deviation of the training
    spectra, bringing target magnitudes into a numerically convenient range. It
    can be provided explicitly or resolved automatically from the training data
    via the ``"auto"`` token.

    Two orthogonal switches control how the factor is shaped. ``per_point``
    selects whether the factor varies across the spectrum (a length-``N`` vector)
    or is a single global scalar shared across all points. ``per_element``
    selects whether a separate factor is used for each target-site element
    (chosen per sample from its atomic number) or a single factor is shared
    across all elements.

    When ``per_element`` is false, ``factor`` is a flat list. When it is true,
    it is a list of rows, one row per entry of ``elements``.

    Zero or near-zero factors (below ``1e-12``) are silently replaced with
    ``1.0``, which is equivalent to skipping scaling at those points. This
    handles flat spectral regions where the per-point
    standard deviation vanishes.

    Args:
        encoding_type: Identifier string for this encoding type.
        factor: Divisor applied during encoding. A flat list when
            ``per_element`` is false, or one row per element (aligned with
            ``elements``) when true.
        per_point: Whether scaling is applied per spectrum point (``True``) or
            globally with a single shared factor (``False``).
        per_element: Whether the factor is selected per target-site element
            (``True``) or shared across all elements (``False``).
        elements: Atomic numbers aligned row-wise with ``factor`` when
            ``per_element`` is true; ignored otherwise.

    Raises:
        ConfigError: If ``factor`` is empty.
    """

    def __init__(
        self,
        encoding_type: str,
        factor: list[float] | list[list[float]],
        per_point: bool,
        per_element: bool,
        elements: list[int] | None,
    ) -> None:
        """Initialize ``ScaleEncoding``."""
        if per_element:
            factor_tensor = build_parameter_rows(encoding_type, "factor", factor)  # type: ignore[arg-type]
        else:
            if not factor:
                raise ConfigError(f"{encoding_type} requires a non-empty factor.")
            factor_tensor = torch.tensor(factor, dtype=torch.float32)

        # Replace near-zero factors with 1.0 (identity scaling) so that flat
        # spectral regions where std \approx 0 do not cause
        # division-by-zero during encoding.
        factor_tensor = torch.where(
            factor_tensor < 1e-12,
            torch.ones_like(factor_tensor),
            factor_tensor,
        )

        self.per_point = per_point
        super().__init__(encoding_type, torch.zeros_like(factor_tensor), factor_tensor, per_element, elements)

    @property
    def signature(self) -> list[Config]:
        """Return the scale-encoding signature.

        Returns:
            Configuration values needed to recreate this encoding.
        """
        sig = super().signature[0]
        values: dict[str, object] = {
            "factor": self.scale.tolist(),
            "per_point": self.per_point,
            "per_element": self.per_element,
        }
        if self.per_element:
            values["elements"] = list(self.elements or [])
        sig.update_with_dict(values)
        return [sig]


@EncodingAutoResolver.register("scale")
def resolve_scale_encoding(item: ConfigRaw, statistics: SpectralStatisticsCollector) -> ConfigRaw:
    """Resolve the scaling factor from the training spectra.

    The factor is the population standard deviation, so encoding divides the
    spectra to unit variance.

    Args:
        item: Raw scale encoding configuration dictionary.
        statistics: Streaming statistics of the training spectra
            (:class:`~xanesnet.serialization.auto_config.statistics.SpectralStatisticsCollector`).

    Returns:
        Mapping with ``factor``.  When ``per_element`` is false this is a
        per-point list (or single-element list when ``per_point`` is false);
        when true it is one row per element together with the resolved
        ``elements``.
    """
    per_point = item.get("per_point", True)
    if item.get("per_element", False):
        requested = item.get("elements")
        elements = [int(z) for z in requested] if isinstance(requested, list) else statistics.sorted_elements()
        factor_rows: list[list[float]] = []
        for z in elements:
            stats = statistics.element_stats(z)
            factor_rows.append(stats.std.tolist() if per_point else [stats.global_std])
        return {"elements": list(elements), "factor": factor_rows}

    stats = statistics.overall
    if per_point:
        return {"factor": stats.std.tolist()}
    return {"factor": [stats.global_std]}
