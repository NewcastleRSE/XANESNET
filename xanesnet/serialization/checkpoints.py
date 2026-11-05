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

"""Checkpoint dataclass for XANESNET."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import Config

# Training-state values stored once per strategy member.
_PER_MEMBER_KEYS = (
    "epochs",
    "optimizer_states",
    "epoch_scheduler_states",
    "warmup_scheduler_states",
)


@dataclass
class Checkpoint:
    """Checkpoint holding model state and optional per-member training state.

    Args:
        signature: Run signature with ``dataset``, ``model``, and ``encodings`` sections.
        model_states: One ``state_dict`` per strategy member.
        training_state: Per-member epoch counts, optimizer and scheduler states, and signatures.
    """

    signature: Config
    model_states: list[Mapping[str, Any]]
    training_state: dict[str, Any] | None = None

    def save(self, path: str | Path) -> Path:
        """Write the minimal checkpoint and, when available, its full-state sibling.

        The full checkpoint is written to ``<stem>_full<suffix>`` in the same directory.

        Args:
            path: Destination of the minimal checkpoint, with a ``.pth`` suffix.

        Returns:
            The ``Path`` of the minimal checkpoint.
        """
        return self._write(path)

    def save_separate(self, path: str | Path) -> list[Path]:
        """Write one minimal/full checkpoint pair per model state.

        Each checkpoint keeps the original run signature and contains one model
        state plus the corresponding per-member training state.

        Args:
            path: Base checkpoint path used to derive each member filename.

        Returns:
            Minimal checkpoint paths written for individual members.
        """
        if len(self.model_states) < 2:
            return []

        path = Path(path)
        saved_paths = []
        for member_index, model_state in enumerate(self.model_states):
            member_training_state = None
            if self.training_state is not None:
                member_training_state = self.training_state.copy()
                for key in _PER_MEMBER_KEYS:
                    if key in member_training_state:
                        member_training_state[key] = [member_training_state[key][member_index]]

            member_checkpoint = Checkpoint(
                signature=self.signature,
                model_states=[model_state],
                training_state=member_training_state,
            )
            member_path = path.with_name(f"{path.stem}_member_{member_index}{path.suffix}")
            saved_paths.append(member_checkpoint._write(member_path))

        return saved_paths

    def _write(self, path: str | Path) -> Path:
        """Write the minimal checkpoint and its optional full-state sibling.

        Args:
            path: Destination path for the minimal checkpoint.

        Returns:
            Path to the written minimal checkpoint.
        """
        path = Path(path)
        state = self.to_state_dict()
        minimal_state = state.copy()
        minimal_state.pop("training_state", None)
        torch.save(minimal_state, path)
        if "training_state" in state:
            torch.save(state, path.with_name(f"{path.stem}_full{path.suffix}"))
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "Checkpoint":
        """Load a checkpoint from a ``.pth`` file.

        Args:
            path: Path to a Checkpoint ``.pth`` file.
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

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize this checkpoint to a plain dictionary suitable for ``torch.save``.

        Returns:
            Checkpoint dict.
        """
        state = {
            "signature": self.signature.as_dict(),
            "model_states": self.model_states,
        }
        if self.training_state is not None:
            state["training_state"] = self.training_state
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Checkpoint":
        """Reconstruct a ``Checkpoint`` from a state dictionary.

        Args:
            state: Dictionary as produced by ``to_state_dict()``.

        Returns:
            A reconstructed ``Checkpoint`` instance.
        """
        return cls(
            signature=Config(state["signature"]),
            model_states=state["model_states"],
            training_state=state.get("training_state"),
        )
