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
    # Has to be Sequence such that it is immutable.
    # Immutability is important such that model_states list
    # can contain 'None'.
    model_states: list[dict]
    signature: dict[str, Any]
    optimizer_states: list[dict] | None = None
    epochs: list[int] | None = None

    def save(self, path: str | Path) -> None:
        """
        Save checkpoint as a .pth file
        """
        torch.save(asdict(self), path)

    def __len__(self) -> int:
        return len(self.model_states)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "Checkpoint":
        """
        Load checkpoint from a file
        """
        data = torch.load(path, map_location=map_location)
        return cls(**data)


def build_checkpoint(
    model_list: list[Model],
    signature: dict[str, Any],
    optimizer_states: list[dict] | None = None,
    epochs: list[int] | None = None,
) -> Checkpoint:
    if len(model_list) == 0:
        raise ValueError("No models. Can not build checkpoint. ")
    else:
        checkpoint = Checkpoint(
            model_states=[model.state_dict() for model in model_list],
            signature=signature,
            optimizer_states=optimizer_states,
            epochs=epochs,
        )
        return checkpoint


def save_checkpoint(
    dst_dir: str | Path,
    checkpoint: Checkpoint,
    name: str | None = None,
) -> Path:
    if not isinstance(dst_dir, Path):
        dst_dir = Path(dst_dir)
    checkpoint.save(dst_dir / f"{name}.pth" if name else dst_dir / "checkpoint.pth")
    return dst_dir / f"{name}.pth" if name else dst_dir / "checkpoint.pth"
