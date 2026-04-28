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

"""Utilities for saving and loading XANESNET model weights."""

from pathlib import Path
from typing import Any

import torch

from xanesnet.models import Model


def save_model(dst_dir: str | Path, model: Model) -> None:
    """Save a single model's weights to ``model_weights.pth`` inside ``dst_dir``.

    Args:
        dst_dir: Directory in which to write the weights file. Created if it
            does not already exist.
        model: The model whose ``state_dict()`` will be saved.
    """
    if not isinstance(dst_dir, Path):
        dst_dir = Path(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dst_dir / "model_weights.pth")


def save_models(dst_dir: str | Path, model_list: list[Model]) -> None:
    """Save one or more models to ``dst_dir``.

    If a single model is provided, its weights are written directly to
    ``dst_dir/model_weights.pth``.  For multiple models, each is saved in its
    own numbered sub-directory (``model_0/``, ``model_1/``, ...).

    Args:
        dst_dir: Destination directory.
        model_list: Non-empty list of models to save.

    Raises:
        ValueError: If ``model_list`` is empty.
    """
    if not isinstance(dst_dir, Path):
        dst_dir = Path(dst_dir)

    if len(model_list) == 0:
        raise ValueError("No models to save.")
    elif len(model_list) == 1:
        save_model(dst_dir, model_list[0])
    else:
        for idx, model in enumerate(model_list):
            model_dir = dst_dir / f"model_{idx}"
            model_dir.mkdir(parents=True, exist_ok=True)
            save_model(model_dir, model)


def load_pretrained_model() -> Any:
    """Load a pre-trained model.

    Note:
        Not implemented yet.

    Returns:
        Loaded model once implemented.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError("Pretrained model loading not implemented yet.")
