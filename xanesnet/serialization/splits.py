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


def load_split_indices(filepath: str | Path) -> list[list[int]]:
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "indices" not in data or not isinstance(data["indices"], dict):
        logging.critical("Invalid split indices file format.")

    raw_indices = data["indices"]

    index_map: dict[int, list[int]] = {}

    # Semantic aliases
    if "train" in raw_indices:
        if "0" in raw_indices:
            logging.critical("Both 'train' and '0' keys present (conflict)")
        index_map[0] = raw_indices["train"]

    if "valid" in raw_indices:
        if "1" in raw_indices:
            logging.critical("Both 'valid' and '1' keys present (conflict)")
        index_map[1] = raw_indices["valid"]

    # Numeric keys
    for key, indices in raw_indices.items():
        if key in {"train", "valid"}:
            continue

        try:
            idx = int(key)
        except ValueError:
            logging.critical(f"Invalid split key '{key}': must be int, 'train', or 'valid'")
            return []  # for type checker

        if idx in index_map:
            logging.critical(f"Duplicate definition for split index {idx}")

        index_map[idx] = indices

    # Validate indices
    for idx, indices in index_map.items():
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            logging.critical(f"Invalid indices for split {idx}: must be a list of integers")

    # Build ordered list
    max_index = max(index_map)
    split_indices_list: list[list[int]] = []

    for i in range(max_index + 1):
        if i not in index_map:
            logging.critical(f"Missing split indices for index {i}")
        split_indices_list.append(index_map[i])

    return split_indices_list


def save_split_indices(
    filepath: str | Path,
    split_indices_list: list[list[int]],
    train_valid_keys: bool = True,
) -> None:
    """
    Save split indices in a human-readable JSON format.
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
