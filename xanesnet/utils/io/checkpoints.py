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

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from xanesnet.models import Model


@dataclass
class Checkpoint:
    model_states: list[dict]
    signature: dict[str, Any]
    optimizer_states: list[dict] | None = None
    epochs: list[int] | None = None

    def save(self, path: str | Path) -> None:
        """
        Save checkpoint as a .pth file
        """
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "Checkpoint":
        """
        Load checkpoint from a file
        """
        data = torch.load(path, map_location=map_location)
        return cls(**data)


def save_checkpoint(
    dst_dir: str | Path,
    model: Model,
    signature: dict[str, Any],
    optimizer_state: dict | None = None,
    epoch: int | None = None,
    name: str | None = None,
) -> None:
    save_checkpoints(
        dst_dir=dst_dir,
        model_list=[model],
        signature=signature,
        optimizer_states=[optimizer_state] if optimizer_state is not None else None,
        epochs=[epoch] if epoch is not None else None,
        name=name,
    )


def save_checkpoints(
    dst_dir: str | Path,
    model_list: list[Model],
    signature: dict[str, Any],
    optimizer_states: list[dict] | None = None,
    epochs: list[int] | None = None,
    name: str | None = None,
) -> None:
    if len(model_list) == 0:
        raise ValueError("No models to save.")
    else:
        checkpoint = Checkpoint(
            model_states=[model.state_dict() for model in model_list],
            signature=signature,
            optimizer_states=optimizer_states,
            epochs=epochs,
        )
        if not isinstance(dst_dir, Path):
            dst_dir = Path(dst_dir)
        checkpoint.save(dst_dir / f"{name}.pth" if name else dst_dir / "checkpoint.pth")
