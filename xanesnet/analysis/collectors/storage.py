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

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class DiskBackedResults(Iterable[dict[str, Any]]):
    """
    Lazy iterable that reads per-sample results from a JSONL file.
    """

    def __init__(self, path: Path, count: int | None = None) -> None:
        self.path = path
        self._count = count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def __len__(self) -> int:
        if self._count is not None:
            return self._count
        meta_path = self.path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self._count = int(meta.get("count", 0))
            return self._count
        return 0


def json_friendly(value: Any) -> Any:
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
