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

"""Abstract base class for XANESNET early-stopping strategies."""

from abc import ABC, abstractmethod
from typing import Any

from xanesnet.checkpointing.checkpointer import state_to_cpu
from xanesnet.components import Optimizer, Scheduler
from xanesnet.models import Model


class EarlyStopper(ABC):
    """Abstract base class for early-stopping strategies.

    Tracks the best observed metric value and optionally snapshots the
    corresponding model, optimizer, and scheduler states for restoration.

    Args:
        early_stopper_type: Registry key identifying this stopper type.
        restore_best: If ``True``, save copies of model, optimizer, and
            scheduler state whenever a new best value is observed.
    """

    def __init__(
        self,
        early_stopper_type: str,
        restore_best: bool,
    ) -> None:
        """Initialize shared early-stopping state."""
        self.early_stopper_type = early_stopper_type
        self.restore_best = restore_best

        self.best_state: dict | None = None
        self.best_training_state: dict[str, Any] | None = None
        self.best_value: float = float("inf")
        self.best_epoch: int = -1

    def start(self) -> None:
        """Reset best-model tracking before a new training call."""
        self.best_state = None
        self.best_training_state = None
        self.best_value = float("inf")
        self.best_epoch = -1

    @abstractmethod
    def step(
        self,
        value: float,
        model: Model,
        epoch: int,
        optimizer: Optimizer,
        epoch_scheduler: Scheduler,
        warmup_scheduler: Scheduler | None,
    ) -> bool:
        """Evaluate whether training should stop after observing ``value``.

        The base implementation updates ``best_value``, ``best_epoch``, and
        (when ``restore_best`` is ``True``) model, optimizer, and scheduler
        snapshots whenever a new minimum is found. Subclasses can call
        ``super().step()`` whenever they decide that the observed value should
        be recorded as the new best, then apply their own stopping criterion.

        Args:
            value: The metric to minimize (e.g. validation loss) at the
                current epoch.
            model: The model being trained; its weights are copied when a
                new best is reached and ``restore_best`` is ``True``.
            epoch: Current epoch index.
            optimizer: Optimizer whose state is copied for a new best.
            epoch_scheduler: Per-epoch scheduler paired with the optimizer.
            warmup_scheduler: Optional per-step warm-up scheduler.

        Returns:
            ``True`` if training should stop, ``False`` otherwise. The base
            implementation always returns ``False``.
        """
        if value < self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                self.best_training_state = state_to_cpu(
                    {
                        "optimizer": optimizer.torch_optimizer.state_dict(),
                        "epoch_scheduler": epoch_scheduler.state_dict(),
                        "warmup_scheduler": None if warmup_scheduler is None else warmup_scheduler.state_dict(),
                    }
                )
        return False

    def restore(
        self,
        model: Model,
        optimizer: Optimizer,
        epoch_scheduler: Scheduler,
        warmup_scheduler: Scheduler | None,
    ) -> tuple[float | None, int | None]:
        """Restore model, optimizer, and schedulers to the best recorded state.

        Only has an effect when ``restore_best`` was set to ``True`` at
        construction and at least one call to ``step`` has been made.

        Args:
            model: Model whose weights will be replaced with the saved snapshot.
            optimizer: Optimizer whose state will be replaced with its snapshot.
            epoch_scheduler: Scheduler whose state will be restored.
            warmup_scheduler: Optional warm-up scheduler to restore.

        Returns:
            A ``(best_value, best_epoch)`` tuple if a best state was
            restored, otherwise ``(None, None)``.
        """
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
            if self.best_training_state is not None:
                optimizer.torch_optimizer.load_state_dict(self.best_training_state["optimizer"])
                epoch_scheduler.load_state_dict(self.best_training_state["epoch_scheduler"])
                if warmup_scheduler is not None:
                    warmup_scheduler.load_state_dict(self.best_training_state["warmup_scheduler"])
            return self.best_value, self.best_epoch

        return None, None
