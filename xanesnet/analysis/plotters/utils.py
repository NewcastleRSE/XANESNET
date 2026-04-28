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

"""Shared helpers for analysis plotters."""

from typing import Any, TypeGuard, cast

from xanesnet.serialization.jsonl_stream import JSONLStream

from ..selectors import Selector

ScalarValue = int | float


def is_scalar(value: Any) -> TypeGuard[ScalarValue]:
    """Return whether ``value`` is a non-boolean Python scalar number.

    Args:
        value: Candidate value from a prediction sample or collector output.

    Returns:
        ``True`` when ``value`` is an ``int`` or ``float``, excluding booleans.
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def collect_scalar_values(selector: Selector, stream: JSONLStream | None) -> dict[str, list[ScalarValue]]:
    """Collect scalar values from selected samples and optional collector outputs.

    Args:
        selector: Selector over prediction samples for one prediction reader and selector pair.
        stream: Optional collector result stream aligned with ``selector``.

    Returns:
        Mapping from scalar value key to values observed for that key.
    """
    values: dict[str, list[ScalarValue]] = {}
    if stream is not None:
        for sel_sample, col_sample in zip(selector, stream):
            for key, val in sel_sample.items():
                if key != "sample_id" and is_scalar(val):
                    values.setdefault(key, []).append(cast(float, val))
            for key, val in col_sample.items():
                if key != "sample_id" and is_scalar(val):
                    values.setdefault(key, []).append(cast(float, val))
    else:
        for sel_sample in selector:
            for key, val in sel_sample.items():
                if key != "sample_id" and is_scalar(val):
                    values.setdefault(key, []).append(cast(float, val))
    return values
