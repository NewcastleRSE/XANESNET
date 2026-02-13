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

from typing import Any, cast

from xanesnet.serialization.jsonl_stream import JSONLStream

from ..selectors import Selector


def is_scalar(value: Any) -> bool:
    """
    Check if a value is a scalar number (int or float, excluding bool).
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def collect_scalar_values(selector: Selector, stream: JSONLStream | None) -> dict[str, list[float]]:
    """
    Walk selector and (optionally) collector stream in parallel.

    Returns ``{key: [values]}`` for every scalar field found.
    """
    values: dict[str, list[float]] = {}
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
