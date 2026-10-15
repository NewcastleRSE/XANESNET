# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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

"""Internal helpers for automatic model and encoding configuration resolution."""

from typing import Any

from ..config import ConfigRaw


def format_value(value: Any) -> str:
    """Recursive formatter for a single value or nested structure.

    Args:
        value: Value to format.

    Returns:
        String representation of the value.
    """
    if isinstance(value, list):
        return _format_list(value)
    if isinstance(value, dict):
        return _format_dict(value)
    return str(value)


def _format_list(lst: list[Any]) -> str:
    """Format a list, truncating when longer than 8 elements.

    Args:
        lst: List to format.

    Returns:
        Compact string representation, truncated when ``len(lst) > 8``.
    """
    if len(lst) <= 8:
        return "[" + ", ".join(format_value(v) for v in lst) + "]"

    head = ", ".join(format_value(v) for v in lst[:3])
    if all(isinstance(v, list) for v in lst):
        # Nested list of rows -- add row count and inner length summary.
        inner_lens = {len(v) for v in lst}
        lens_str = f"inner_len={inner_lens.pop()}" if len(inner_lens) == 1 else f"inner_lens={sorted(inner_lens)}"
        return f"[{head}, ...]  (rows={len(lst)}, {lens_str})"
    return f"[{head}, ...]  (len={len(lst)})"


def _format_dict(dct: dict[str, Any]) -> str:
    """Format a dict, truncating long values.

    Args:
        dct: Dictionary to format.

    Returns:
        Compact string representation, truncated when ``len(dct) > 4``.
    """
    if len(dct) <= 4:
        items = ", ".join(f"{k}={format_value(v)}" for k, v in dct.items())
        return "{" + items + "}"
    keys = list(dct.keys())
    items = ", ".join(f"{k}={format_value(dct[k])}" for k in keys[:3])
    return "{" + f"{items}, ...  (keys={len(dct)})" + "}"


def requested_auto_fields(model_config: ConfigRaw) -> set[str]:
    """Return top-level config fields whose value is ``"auto"``.

    Args:
        model_config: Raw model or encoding configuration dictionary.

    Returns:
        Set of top-level field names requesting automatic resolution.
    """
    return {key for key, value in model_config.items() if isinstance(value, str) and value.lower() == "auto"}
