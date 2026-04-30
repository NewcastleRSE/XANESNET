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

"""Model checkpoint persistence utilities for XANESNET training runs."""

from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

from xanesnet.models import Model
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config


class Checkpointer:
    """Saves periodic model and optimizer checkpoints during training.

    When ``save_dir`` or ``save_interval`` is ``None`` the checkpointer is
    inactive and all methods are no-ops. Active checkpointers store CPU copies
    of model and optimizer states so multi-model checkpoints do not retain old
    GPU tensors in memory.

    Args:
        save_dir: Directory where checkpoint files are written.
        save_interval: Save a checkpoint every this many epochs. Must be a
            positive integer when the checkpointer is active.
        model_signature: Model hyperparameter dict embedded in every checkpoint.
            Required when the checkpointer is active (i.e. both ``save_dir``
            and ``save_interval`` are provided).

    Raises:
        ValueError: If the checkpointer is active and ``save_interval <= 0``
            or ``model_signature`` is ``None``.
    """

    def __init__(
        self,
        save_dir: str | Path | None,
        save_interval: int | None,
        model_signature: Config | None,
    ) -> None:
        """Initialize ``Checkpointer``."""
        self.active = save_dir is not None and save_interval is not None

        if self.active:
            assert save_dir is not None and save_interval is not None
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

            if save_interval <= 0:
                raise ValueError(f"save_interval must be positive, got {save_interval}")
            self.save_interval = save_interval

            self.model_counter = -1
            self.checkpoint_counter = -1

            if model_signature is None:
                raise ValueError("model_signature is required when the checkpointer is active")
            # Create checkpoint with empty lists
            self._checkpoint = Checkpoint(model_states=[], signature=model_signature, optimizer_states=[], epochs=[])

    def step(self, epoch: int, model: Model, optimizer: optim.Optimizer) -> tuple[bool, str]:
        """Conditionally save a checkpoint for the current epoch.

        A checkpoint is written when ``epoch % save_interval == 0``. The new
        state always overwrites the last slot in the checkpoint's model-state,
        optimizer-state, and epoch sequences.

        Args:
            epoch: Current training epoch.
            model: Model whose ``state_dict`` will be persisted.
            optimizer: Optimizer whose ``state_dict`` will be persisted.

        Returns:
            ``(saved, checkpoint_name)`` - ``saved`` is ``True`` when a file
            was written; ``checkpoint_name`` is the file basename or ``""``
            when nothing was saved.
        """
        if self.active and epoch % self.save_interval == 0:
            return self.save_checkpoint(epoch, model, optimizer)

        return False, ""

    def save_checkpoint(self, epoch: int, model: Model, optimizer: optim.Optimizer) -> tuple[bool, str]:
        """Unconditionally write a checkpoint for the current epoch.

        Updates the last slot in the checkpoint's state sequences and writes the
        checkpoint file to disk. Requires :meth:`new_model` to have been called
        at least once so that the state lists are non-empty.

        Args:
            epoch: Current training epoch.
            model: Model whose ``state_dict`` will be persisted.
            optimizer: Optimizer whose ``state_dict`` will be persisted.

        Returns:
            ``(True, checkpoint_name)`` on success, or ``(False, "")`` when
            the checkpointer is inactive.

        Raises:
            ValueError: If :meth:`new_model` has not been called yet.
        """
        if not self.active:
            return False, ""

        if self.model_counter < 0 or len(self._checkpoint.model_states) == 0:
            raise ValueError("new_model() must be called before save_checkpoint()")

        self.checkpoint_counter += 1

        self._checkpoint.model_states[-1] = _state_to_cpu(model.state_dict())
        assert self._checkpoint.optimizer_states is not None
        assert self._checkpoint.epochs is not None
        self._checkpoint.optimizer_states[-1] = _state_to_cpu(optimizer.state_dict())
        self._checkpoint.epochs[-1] = epoch

        checkpoint_name = f"checkpoint_{self.model_counter}_{epoch}.pth"
        self._checkpoint.save(self.save_dir / checkpoint_name)

        return True, checkpoint_name

    def new_model(self) -> int:
        """Register a new model slot in the checkpoint and reset the step counter.

        Appends empty entries to the model-state, optimizer-state, and epoch
        sequences so that subsequent :meth:`save_checkpoint` calls can write
        into the new slot.

        Returns:
            Total number of model slots now registered in the checkpoint, or
            ``0`` when the checkpointer is inactive.
        """
        if not self.active:
            return 0

        self.model_counter += 1
        self.checkpoint_counter = 0

        assert self._checkpoint.optimizer_states is not None
        assert self._checkpoint.epochs is not None

        self._checkpoint.model_states.append({})
        self._checkpoint.optimizer_states.append({})
        self._checkpoint.epochs.append(-1)

        return len(self._checkpoint)


def _state_to_cpu(value: Any) -> Any:
    """Recursively copy tensors in a state object to CPU.

    Args:
        value: A tensor, mapping, sequence, or scalar value from a model or
            optimizer state dictionary.

    Returns:
        A structurally equivalent object with all tensors detached, cloned, and
        moved to CPU.
    """
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _state_to_cpu(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(subvalue) for subvalue in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(subvalue) for subvalue in value)
    return value
