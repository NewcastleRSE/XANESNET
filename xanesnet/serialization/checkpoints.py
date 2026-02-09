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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from xanesnet.models import Model

from .config import Config


@dataclass
class Checkpoint:
    model_states: list[dict]
    signature: Config
    optimizer_states: list[dict] | None = None
    epochs: list[int] | None = None

    def save(self, path: str | Path) -> Path:
        """
        Save checkpoint as a .pth file
        """
        path = Path(path)

        if path.suffix != ".pth":
            raise ValueError(f"Checkpoint path must end with .pth. Got: {path}")

        if not path.parent.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {path.parent}")

        torch.save(self.to_state_dict(), path)

        return path

    def __len__(self) -> int:
        return len(self.model_states)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "Checkpoint":
        """
        Load checkpoint from a file
        """
        state = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )

        return cls.from_state_dict(state)

    @classmethod
    def build(
        cls,
        model_list: list[Model],
        signature: Config,
        optimizer_states: list[dict] | None = None,
        epochs: list[int] | None = None,
    ) -> "Checkpoint":
        if len(model_list) == 0:
            raise ValueError("No models. Can not build checkpoint. ")
        else:
            checkpoint = cls(
                model_states=[model.state_dict() for model in model_list],
                signature=signature,
                optimizer_states=optimizer_states,
                epochs=epochs,
            )
            return checkpoint

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "model_states": self.model_states,
            "optimizer_states": self.optimizer_states,
            "epochs": self.epochs,
            "signature": self.signature.as_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Checkpoint":
        return cls(
            model_states=state["model_states"],
            optimizer_states=state.get("optimizer_states"),
            epochs=state.get("epochs"),
            signature=Config(state["signature"]),
        )
