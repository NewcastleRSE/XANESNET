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

"""Dataset split index loading and saving for XANESNET."""

import json
from pathlib import Path

from xanesnet.utils.exceptions import ResourceError


def load_splits(filepath: str | Path, model_index: int | None = None) -> list[list[list[int]]]:
    """Load split indices for all models or a selected model.

    Args:
        filepath: Path to the model-first split-index file.
        model_index: Optional zero-based model position to load by itself.

    Returns:
        Ordered split lists for every model, or a one-model list when ``model_index`` is given.

    Raises:
        ResourceError: If the file format or split indices are invalid, or ``model_index`` is out of range.
    """
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ResourceError("Invalid model splits file format.")

    model_splits: list[list[list[int]]] = []
    for model_number, raw_splits in enumerate(data["models"]):
        if not isinstance(raw_splits, dict):
            raise ResourceError(f"Model split entry {model_number} must be a dict.")
        unexpected_keys = raw_splits.keys() - {"train", "valid"}
        if unexpected_keys:
            raise ResourceError(f"Unexpected split keys: {sorted(unexpected_keys)}")
        if raw_splits and "train" not in raw_splits:
            raise ResourceError(f"Model split entry {model_number} is missing 'train'.")

        splits = []
        for split_name in ("train", "valid"):
            if split_name not in raw_splits:
                continue
            indices = raw_splits[split_name]
            if not isinstance(indices, list) or not all(
                isinstance(index, int) and not isinstance(index, bool) for index in indices
            ):
                raise ResourceError(f"Invalid indices for split '{split_name}': must be a list of integers")
            splits.append(indices)
        model_splits.append(splits)

    if model_index is None:
        return model_splits
    if not isinstance(model_index, int) or isinstance(model_index, bool) or not 0 <= model_index < len(model_splits):
        raise ResourceError(f"Model index {model_index} is out of range.")
    return [model_splits[model_index]]


def save_splits(filepath: str | Path, model_splits: list[list[list[int]]]) -> None:
    """Save dataset split indices for each model to a JSON file.

    Args:
        filepath: Destination path for the JSON file.  Parent directories are created automatically.
        model_splits: One ordered list of split index lists per model.
            Positions 0 and 1 are written as ``"train"`` and ``"valid"``.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    split_names = ("train", "valid")
    if any(len(model_indices) > len(split_names) for model_indices in model_splits):
        raise ValueError("Each model may have at most train and validation splits.")

    serialized_models = []
    for model_indices in model_splits:
        serialized_models.append(dict(zip(split_names, model_indices)))

    with filepath.open("w", encoding="utf-8") as f:
        json.dump({"models": serialized_models}, f, indent=2)
