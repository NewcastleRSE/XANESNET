# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Variance-preserving scale factors and JSON-loading utilities for GemNet calibration."""

import json
import logging
from pathlib import Path

import torch


class ScaleFactor(torch.nn.Module):
    """Scalar variance-preserving factor.

    The default value is 1.0 (identity / no scaling). A model loaded with
    ``scale_file=None`` therefore behaves as if no scaling were applied.
    Pre-fitted values are produced offline by
    ``scripts/gemnet_scale_fitting.py`` and persisted either via
    :func:`load_scales_json` (loading from the JSON the fitter writes) or
    automatically as part of the model's ``state_dict``.

    The ``ref`` argument is accepted (and ignored) so call sites match the
    fairchem-style ``forward(x, ref=...)`` signature observed by the
    offline fitter via forward hooks.

    Args:
        name: Optional human-readable label used as the JSON key when scales
            are fitted or loaded.
    """

    scale_factor: torch.Tensor

    def __init__(self, name: str | None = None) -> None:
        """Initialize ``ScaleFactor``."""
        super().__init__()
        self.name = name
        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x: torch.Tensor, ref: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the configured scale factor.

        Args:
            x: Input tensor to scale.
            ref: Ignored reference tensor accepted for API compatibility.

        Returns:
            ``x`` multiplied by the stored scale factor.
        """
        return x * self.scale_factor


class ScalingFactor(ScaleFactor):
    """Legacy GemNet API shim: ``forward(x_ref, y)`` with the same semantics as
    :class:`ScaleFactor` (scale ``y``, observe ``x_ref`` as input).

    The ``scale_file`` and ``device`` constructor arguments are accepted for
    backwards-compat with the pre-existing GemNet layer call sites but are
    ignored - scales are loaded by the model itself via
    :func:`load_scales_json` (or via ``state_dict`` round-trip).

    Args:
        scale_file: Ignored legacy argument kept for constructor compatibility.
        name: Optional human-readable label used as the JSON key.
        device: Optional device used to place the scale parameter.
    """

    def __init__(
        self,
        scale_file: str | None = None,
        name: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize ``ScalingFactor``."""
        super().__init__(name=name)
        if device is not None:
            self.scale_factor.data = self.scale_factor.data.to(device)

    def forward(self, x_ref: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply the configured scale factor.

        Args:
            x_ref: Ignored reference tensor accepted for API compatibility.
            y: Tensor to scale.

        Returns:
            ``y`` multiplied by the stored scale factor.
        """
        return super().forward(y, ref=x_ref)


def collect_scale_factors(model: torch.nn.Module) -> dict[str, ScaleFactor]:
    """Return all :class:`ScaleFactor` submodules keyed by their dotted name.

    Args:
        model: Module tree to inspect.

    Returns:
        Dictionary mapping dotted module paths to :class:`ScaleFactor` instances.
    """
    return {name: m for name, m in model.named_modules() if isinstance(m, ScaleFactor)}


def load_scales_json(model: torch.nn.Module, path: str | Path, *, strict: bool = False) -> int:
    """Load fitted scale-factor values from a JSON file produced by
    ``scripts/gemnet_scale_fitting.py`` into ``model``.

    Missing keys are tolerated unless ``strict=True``; the corresponding
    :class:`ScaleFactor` submodules remain at 1.0 (identity).

    Args:
        model: Module whose :class:`ScaleFactor` parameters to update.
        path: Path to the JSON file produced by the scale fitter.
        strict: If ``True``, raise :class:`ValueError` when any
            :class:`ScaleFactor` key is absent from the file.

    Returns:
        Number of scale factors successfully loaded.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not contain a JSON object, or if
            ``strict=True`` and a scale key is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scale file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Scale file {path} does not contain a JSON object")

    factors = collect_scale_factors(model)
    loaded = 0
    missing_in_file: list[str] = []
    for name, sf in factors.items():
        if name in payload:
            with torch.no_grad():
                sf.scale_factor.fill_(float(payload[name]))
            loaded += 1
        else:
            missing_in_file.append(name)

    extra_in_file = [k for k in payload if k not in factors]

    if loaded:
        logging.info("Loaded %d scale factors from %s", loaded, path)
    if missing_in_file:
        msg = f"Scale file {path} missing {len(missing_in_file)} factors: {missing_in_file}"
        if strict:
            raise ValueError(msg)
        logging.warning(msg)
    if extra_in_file:
        logging.warning("Scale file %s has %d unknown keys: %s", path, len(extra_in_file), extra_in_file)

    return loaded
