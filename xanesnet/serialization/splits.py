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

"""Dataset split index loading and saving for XANESNET."""

import json
from pathlib import Path

from xanesnet.utils.exceptions import ResourceError


def load_split_indices(filepath: str | Path) -> list[list[int]]:
    """Load dataset split indices from a JSON file.

    The file must have an ``indices`` dict at the top level.  Keys can be the
    string literals ``"train"`` and ``"valid"`` (mapped to positions 0 and 1
    respectively) or integer strings.

    Args:
        filepath: Path to the JSON split-index file.

    Returns:
        An ordered ``list[list[int]]`` where each inner list contains the
        dataset indices for that split position.

    Raises:
        ResourceError: If the file format is invalid, required keys are missing,
            duplicate or conflicting key definitions are present, or any split's
            indices are not a list of integers.
    """
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "indices" not in data or not isinstance(data["indices"], dict):
        raise ResourceError("Invalid split indices file format.")

    raw_indices = data["indices"]

    index_map: dict[int, list[int]] = {}

    # Semantic aliases
    if "train" in raw_indices:
        if "0" in raw_indices:
            raise ResourceError("Both 'train' and '0' keys present (conflict)")
        index_map[0] = raw_indices["train"]

    if "valid" in raw_indices:
        if "1" in raw_indices:
            raise ResourceError("Both 'valid' and '1' keys present (conflict)")
        index_map[1] = raw_indices["valid"]

    # Numeric keys
    for key, indices in raw_indices.items():
        if key in {"train", "valid"}:
            continue

        try:
            idx = int(key)
        except ValueError:
            raise ResourceError(f"Invalid split key '{key}': must be int, 'train', or 'valid'")

        if idx in index_map:
            raise ResourceError(f"Duplicate definition for split index {idx}")

        index_map[idx] = indices

    # Validate indices
    for idx, indices in index_map.items():
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            raise ResourceError(f"Invalid indices for split {idx}: must be a list of integers")

    # Build ordered list
    max_index = max(index_map)
    split_indices_list: list[list[int]] = []

    for i in range(max_index + 1):
        if i not in index_map:
            raise ResourceError(f"Missing split indices for index {i}")

        split_indices_list.append(index_map[i])

    return split_indices_list


def save_split_indices(
    filepath: str | Path,
    split_indices_list: list[list[int]],
    train_valid_keys: bool = True,
) -> None:
    """Save dataset split indices to a JSON file.

    Args:
        filepath: Destination path for the JSON file.  Parent directories are
            created automatically.
        split_indices_list: Ordered list of index lists, one per split.
        train_valid_keys: If ``True`` (default), positions 0 and 1 are written
            with the human-readable keys ``"train"`` and ``"valid"`` instead of
            ``"0"`` and ``"1"``.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[int]] = {}

    for i, indices in enumerate(split_indices_list):
        key = "train" if i == 0 and train_valid_keys else "valid" if i == 1 and train_valid_keys else str(i)
        splits[key] = indices

    data = {"indices": splits}

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
