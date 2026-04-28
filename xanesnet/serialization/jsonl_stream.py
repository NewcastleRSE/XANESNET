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

"""Lazy JSONL stream reader and JSON serialization helpers for XANESNET."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class JSONLStream(Iterable[dict[str, Any]]):
    """Lazy iterable that reads JSONL lines from a file on demand.

    Args:
        path: Path to the ``.jsonl`` file to read.
        count: Pre-known number of lines.  When ``None``, the count is
            resolved lazily from a ``.meta.json`` sidecar or by scanning
            the file.
    """

    def __init__(self, path: Path, count: int | None = None) -> None:
        """Initialize a lazy JSONL reader.

        Args:
            path: Path to the ``.jsonl`` file to read.
            count: Optional precomputed record count.
        """
        self.path = path
        self._count = count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield each non-empty line of the JSONL file as a parsed dict.

        Returns:
            Iterator over decoded JSON objects, one per non-empty line.
        """
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def __len__(self) -> int:
        """Return the number of records in the stream.

        If a count was not provided at construction time, the value is resolved
        from a ``.meta.json`` sidecar file (key ``count``). If no sidecar
        exists, all lines are counted by scanning the file.

        Returns:
            Total number of non-empty lines / records.
        """
        if self._count is not None:
            return self._count
        meta_path = self.path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self._count = int(meta.get("count", 0))
            return self._count
        logging.warning(f"Count not provided and no meta file found for {self.path}. Counting lines...")
        count = 0
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        self._count = count
        return count


def json_friendly(value: Any) -> Any:
    """Convert a value to a JSON-serializable form.

    Conversion priority:
    1. ``None`` and JSON primitives (``str``, ``int``, ``float``, ``bool``) are
       returned as-is.
    2. ``Path`` objects are converted to strings.
    3. Dicts and lists/tuples are recursed into.
    4. Objects with an ``.item()`` method (e.g. numpy scalars) are unwrapped.
    5. Objects with a ``.tolist()`` method (e.g. numpy/torch arrays) are
       converted to nested Python lists.
    6. Anything else is coerced to ``str``.

    Args:
        value: Any Python value to make JSON-friendly.

    Returns:
        A JSON-serializable Python object.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_friendly(val) for val in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)
