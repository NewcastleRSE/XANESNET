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

"""Checkpoint collection and persistence for XANESNET training runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config

# Needed to avoid circular imports with the Trainer class.
if TYPE_CHECKING:
    from xanesnet.runners.trainers.base import Trainer


class Checkpointer:
    """Collect and write full checkpoints for a training run.

    Periodic writes are enabled only when both ``save_dir`` and ``save_interval`` are provided.
    Snapshot collection and final checkpoint saving remain available when periodic writes are disabled.

    Args:
        save_dir: Directory where checkpoint files are written, or ``None`` to disable periodic writes.
        save_interval: Zero-based epoch interval for periodic writes, or ``None`` to disable them.
        signature: Run signature stored at the top level of each checkpoint.
        trainer_provider: Callable returning each current trainer, or ``None`` after a trainer is released.
    """

    def __init__(
        self,
        save_dir: str | Path | None,
        save_interval: int | None,
        signature: Config,
        trainer_provider: Callable[[], Sequence[Trainer | None]],
    ) -> None:
        """Initialize ``Checkpointer``."""
        self.save_dir = None if save_dir is None else Path(save_dir)
        self.save_interval = save_interval
        self.signature = signature
        self.trainer_provider = trainer_provider

        self._optimizer_signature: dict[str, Any] | None = None
        self._epoch_scheduler_signature: dict[str, Any] | None = None
        self._warmup_scheduler_signature: dict[str, Any] | None = None

        self._finished_states: dict[int, dict[str, Any]] = {}
        self._member_index = 0

        if self.save_dir is not None and self.save_interval is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def finish_member(self, trainer: Trainer) -> None:
        """Cache the current member's state and advance to the next member.

        Args:
            trainer: Trainer that completed this member.
        """
        self._finished_states[self._member_index] = self._collect_member_state(trainer)
        self._member_index += 1

    def snapshot(self) -> Checkpoint:
        """Build a full checkpoint from finished and active strategy members.

        The trainer provider must return one entry per strategy member.
        A released trainer is represented by its cached state from ``finish_member``.

        Returns:
            Checkpoint containing model states and per-member training states.
        """
        trainers = self.trainer_provider()
        member_states = []
        for index, trainer in enumerate(trainers):
            if trainer is None or index in self._finished_states:
                member_states.append(self._finished_states[index])
            else:
                member_states.append(self._collect_member_state(trainer))

        return Checkpoint(
            signature=self.signature,
            model_states=[member["model_state"] for member in member_states],
            training_state={
                "epochs": [member["epochs"] for member in member_states],
                "optimizer_signature": self._optimizer_signature,
                "optimizer_states": [member["optimizer_state"] for member in member_states],
                "epoch_scheduler_signature": self._epoch_scheduler_signature,
                "epoch_scheduler_states": [member["epoch_scheduler_state"] for member in member_states],
                "warmup_scheduler_signature": self._warmup_scheduler_signature,
                "warmup_scheduler_states": [member["warmup_scheduler_state"] for member in member_states],
            },
        )

    def _collect_member_state(self, trainer: Trainer) -> dict[str, Any]:
        """Collect one trainer's state and cache the first component signatures.

        Args:
            trainer: Trainer whose current state is collected.

        Returns:
            Mapping containing the model state, completed epochs, optimizer state, and scheduler states.
        """
        if self._optimizer_signature is None:
            self._optimizer_signature = trainer.optimizer.signature.as_dict()
            self._epoch_scheduler_signature = trainer.epoch_lr_scheduler.signature.as_dict()
            self._warmup_scheduler_signature = (
                None if trainer.warmup_lr_scheduler is None else trainer.warmup_lr_scheduler.signature.as_dict()
            )

        warmup_state = None if trainer.warmup_lr_scheduler is None else trainer.warmup_lr_scheduler.state_dict()
        return _state_to_cpu(
            {
                "model_state": trainer.model.state_dict(),
                "epochs": trainer.completed_epochs,
                "optimizer_state": trainer.optimizer.state_dict(),
                "epoch_scheduler_state": trainer.epoch_lr_scheduler.state_dict(),
                "warmup_scheduler_state": warmup_state,
            }
        )

    def save(self, path: str | Path, save_separate: bool = True) -> Path:
        """Write the checkpoint.

        Args:
            path: Destination of the minimal checkpoint, with a ``.pth`` suffix.
            save_separate: Also write one minimal/full checkpoint pair per member when multiple models are present.

        Returns:
            The ``Path`` of the minimal checkpoint.
        """
        checkpoint = self.snapshot()
        saved_path = checkpoint.save(path)
        if save_separate:
            checkpoint.save_separate(path)
        return saved_path

    def step(self, epoch: int, force_save: bool = False) -> str | None:
        """Save a recovery checkpoint on interval or when explicitly forced.

        Args:
            epoch: Zero-based index of the epoch that just completed.
            force_save: Save even when ``epoch`` is outside the configured interval.

        Returns:
            Basename of the written minimal checkpoint, or ``None`` when nothing was saved.
        """
        if self.save_dir is None or self.save_interval is None:
            return None

        if not force_save and epoch % self.save_interval != 0:
            return None

        checkpoint_name = f"checkpoint_{epoch}_member_{self._member_index}.pth"
        self.snapshot().save(self.save_dir / checkpoint_name)

        return checkpoint_name


def _state_to_cpu(value: Any) -> Any:
    """Recursively detach, clone, and move tensors in nested state to CPU.

    Args:
        value: Tensor or nested mapping, list, tuple, or other state value.

    Returns:
        The value with nested tensors detached, cloned, and moved to CPU. Nested mappings,
        lists, and tuples are reconstructed; other values are returned unchanged.
    """
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _state_to_cpu(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(subvalue) for subvalue in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(subvalue) for subvalue in value)
    return value
