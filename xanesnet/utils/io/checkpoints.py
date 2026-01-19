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
from typing import Any, Dict, Optional

import torch


@dataclass
class Checkpoint:
    model_state: Dict[str, Any]
    metadata: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    epoch: Optional[int] = None

    def save(self, path):
        """Save checkpoint as a .pth file"""
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        """Load checkpoint from a file"""
        data = torch.load(path, map_location=map_location)
        return cls(**data)


def save_checkpoint(dst_dir, model, metadata, optimizer_state=None, epoch=None, name=None):
    checkpoint = Checkpoint(
        model_state=model.state_dict(),
        metadata=metadata,
        optimizer_state=optimizer_state,
        epoch=epoch,
    )
    if name:
        checkpoint.save(dst_dir / f"{name}.pth")
    else:
        checkpoint.save(dst_dir / "checkpoint.pth")


def save_checkpoints(dst_dir, model_list, metadata, optimizer_states=None, epochs=None, name=None):
    if len(model_list) == 0:
        raise ValueError("No models to save.")
    elif len(model_list) == 1:
        save_checkpoint(
            dst_dir,
            model_list[0],
            metadata,
            optimizer_state=optimizer_states[0] if optimizer_states else None,
            epoch=epochs[0] if epochs else None,
            name=name,
        )
    else:
        for idx, model in enumerate(model_list):
            save_checkpoint(
                dst_dir,
                model,
                metadata,
                optimizer_state=optimizer_states[idx] if optimizer_states else None,
                epoch=epochs[idx] if epochs else None,
                name=f"{idx}_{name}" if name else f"{idx}",
            )
