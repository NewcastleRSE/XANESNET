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

import json
import logging
from pathlib import Path

import torch


class ScaleFactor(torch.nn.Module):
    """
    Scalar variance-preserving factor.

    The default value is 1.0 (identity / no scaling). A model loaded with
    ``scale_file=None`` therefore behaves as if no scaling were applied.
    Pre-fitted values are produced offline by
    ``scripts/gemnet_scale_fitting.py`` and persisted either via
    :func:`load_scales_json` (loading from the JSON the fitter writes) or
    automatically as part of the model's ``state_dict``.

    The ``ref`` argument is accepted (and ignored) so call sites match the
    fairchem-style ``forward(x, ref=...)`` signature observed by the
    offline fitter via forward hooks.
    """

    scale_factor: torch.Tensor

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name
        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x: torch.Tensor, ref: torch.Tensor | None = None) -> torch.Tensor:
        return x * self.scale_factor


def collect_scale_factors(model: torch.nn.Module) -> dict[str, ScaleFactor]:
    """
    Return all :class:`ScaleFactor` submodules keyed by their dotted name.
    """
    return {name: m for name, m in model.named_modules() if isinstance(m, ScaleFactor)}


def load_scales_json(model: torch.nn.Module, path: str | Path, *, strict: bool = False) -> int:
    """
    Load fitted scale-factor values from a JSON file produced by
    ``scripts/gemnet_scale_fitting.py`` into ``model``.

    Returns the number of factors successfully loaded. Missing keys are
    tolerated (unless ``strict=True``); the corresponding ``ScaleFactor``
    submodules remain at 1.0 (identity).
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
