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

from pathlib import Path
from typing import Any

import torch

from xanesnet.models import Model


def save_model(dst_dir: str | Path, model: Model) -> None:
    if not isinstance(dst_dir, Path):
        dst_dir = Path(dst_dir)

    torch.save(model.state_dict(), dst_dir / "model_weights.pth")


def save_models(dst_dir: str | Path, model_list: list[Model]) -> None:
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
    raise NotImplementedError("Pretrained model loading not implemented yet.")
