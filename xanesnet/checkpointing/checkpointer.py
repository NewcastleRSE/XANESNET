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

import torch.optim as optim

from xanesnet.models import Model
from xanesnet.serialization import Checkpoint, save_checkpoint


class Checkpointer:
    def __init__(
        self,
        save_dir: str | Path | None,
        save_interval: int | None,
        model_signature: dict[str, Any] | None,
    ) -> None:
        self.active = save_dir is not None and save_interval is not None

        if self.active:
            assert save_dir is not None and save_interval is not None
            self.save_dir = Path(save_dir)
            self.save_interval = save_interval

            self.model_counter = -1
            self.checkpoint_counter = -1

            assert model_signature is not None
            # Create checkpoint with empty lists
            self._checkpoint = Checkpoint(model_states=[], signature=model_signature, optimizer_states=[], epochs=[])

    def step(self, epoch: int, model: Model, optimizer: optim.Optimizer) -> tuple[bool, str]:
        """
        Main step function for the checkpointer that checks if a new checkpoint should be saved.
        If epoch % save_interval == 0 then save model, optimizer, and epoch in checkpoint.
        Always overwrites last model_state, optimizer_state, and epoch in checkpoint Sequences.
        """
        if self.active and epoch % self.save_interval == 0:
            return self.save_checkpoint(epoch, model, optimizer)

        return False, ""

    def save_checkpoint(self, epoch: int, model: Model, optimizer: optim.Optimizer) -> tuple[bool, str]:
        if not self.active:
            return False, ""

        self.checkpoint_counter += 1

        self._checkpoint.model_states[-1] = model.state_dict()
        assert self._checkpoint.optimizer_states is not None
        assert self._checkpoint.epochs is not None
        self._checkpoint.optimizer_states[-1] = optimizer.state_dict()
        self._checkpoint.epochs[-1] = epoch

        checkpoint_name = f"checkpoint_{self.model_counter}_{epoch}"
        save_checkpoint(self.save_dir, self._checkpoint, checkpoint_name)

        return True, checkpoint_name

    def new_model(self) -> int:
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
