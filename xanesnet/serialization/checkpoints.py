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

"""Checkpoint dataclass and helpers for XANESNET model persistence."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from xanesnet.models import Model

from .config import Config


@dataclass
class Checkpoint:
    """Serializable snapshot of one or more trained models.

    Args:
        model_states: List of ``model.state_dict()`` dicts, one per trained model.
        signature: The ``Config`` used to produce this checkpoint.
        optimizer_states: Optional list of ``optimizer.state_dict()`` dicts, one per model.
        epochs: Optional list of epoch counts, one per model.
    """

    model_states: list[dict]
    signature: Config
    optimizer_states: list[dict] | None = None
    epochs: list[int] | None = None

    def save(self, path: str | Path) -> Path:
        """Save the checkpoint to a ``.pth`` file.

        Args:
            path: Destination file path. Must have a ``.pth`` suffix.

        Returns:
            The resolved ``Path`` to the saved file.

        Raises:
            ValueError: If ``path`` does not end with ``.pth``.
            FileNotFoundError: If the parent directory does not exist.
        """
        path = Path(path)

        if path.suffix != ".pth":
            raise ValueError(f"Checkpoint path must end with .pth. Got: {path}")

        if not path.parent.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {path.parent}")

        torch.save(self.to_state_dict(), path)

        return path

    def __len__(self) -> int:
        """Return the number of models stored in this checkpoint.

        Returns:
            Number of serialized model state dictionaries.
        """
        return len(self.model_states)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "Checkpoint":
        """Load a checkpoint from a ``.pth`` file.

        Args:
            path: Path to the ``.pth`` file.
            map_location: Device string passed to ``torch.load`` (default ``"cpu"``).

        Returns:
            A ``Checkpoint`` instance reconstructed from the saved state dict.
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
        """Build a ``Checkpoint`` from a list of trained models.

        Args:
            model_list: Non-empty list of trained ``Model`` instances.
            signature: The ``Config`` used for this training run.
            optimizer_states: Optional optimizer state dicts, one per model.
            epochs: Optional epoch counts, one per model.

        Returns:
            A new ``Checkpoint`` containing the state dicts of all models.

        Raises:
            ValueError: If ``model_list`` is empty, or if ``optimizer_states``
                or ``epochs`` are provided with lengths that do not match the
                number of models.
        """
        if len(model_list) == 0:
            raise ValueError("No models. Can not build checkpoint.")
        if optimizer_states is not None and len(optimizer_states) != len(model_list):
            raise ValueError(
                "optimizer_states must contain one state dict per model. "
                f"Got {len(optimizer_states)} for {len(model_list)} models."
            )
        if epochs is not None and len(epochs) != len(model_list):
            raise ValueError(
                "epochs must contain one epoch value per model. "
                f"Got {len(epochs)} for {len(model_list)} models."
            )
        checkpoint = cls(
            model_states=[model.state_dict() for model in model_list],
            signature=signature,
            optimizer_states=optimizer_states,
            epochs=epochs,
        )
        return checkpoint

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize this checkpoint to a plain dictionary suitable for ``torch.save``.

        Returns:
            Dict with keys ``model_states``, ``optimizer_states``, ``epochs``, and ``signature``.
        """
        return {
            "model_states": self.model_states,
            "optimizer_states": self.optimizer_states,
            "epochs": self.epochs,
            "signature": self.signature.as_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Checkpoint":
        """Reconstruct a ``Checkpoint`` from a state dictionary.

        Args:
            state: Dictionary as produced by ``to_state_dict()``.

        Returns:
            A reconstructed ``Checkpoint`` instance.
        """
        return cls(
            model_states=state["model_states"],
            optimizer_states=state.get("optimizer_states"),
            epochs=state.get("epochs"),
            signature=Config(state["signature"]),
        )
