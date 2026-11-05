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

"""Learning-rate scheduler and registry for XANESNET training."""

# TODO: Maybe there is a better way to implement this

from typing import Any

import torch.optim as optim

from xanesnet.components.optim import Optimizer
from xanesnet.serialization.config import Config
from xanesnet.utils.registry import Registry


class Scheduler(optim.lr_scheduler.LRScheduler):
    """Base class for XANESNET learning-rate schedulers backed by PyTorch."""

    lr_scheduler_type: str
    _signature: Config

    @property
    def signature(self) -> Config:
        """Return the scheduler type required to reconstruct this component.

        Returns:
            Minimal configuration containing the scheduler registry key.
        """
        return self._signature

    @property
    def torch_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """Return this scheduler for framework integrations.

        Returns:
            This native PyTorch scheduler instance.
        """
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state without XANESNET signature metadata.

        Returns:
            State dictionary containing only PyTorch scheduler state.
        """
        state = super().state_dict()
        state.pop("_signature", None)
        state.pop("lr_scheduler_type", None)
        return state

    def _set_scheduler_type(self, lr_scheduler_type: str) -> None:
        """Store the configured scheduler type and its reconstruction signature.

        Args:
            lr_scheduler_type: Configured name identifying this scheduler type.
        """
        self.lr_scheduler_type = lr_scheduler_type
        self._signature = Config({"lr_scheduler_type": self.lr_scheduler_type})


LRSchedulerRegistry: Registry[type[Scheduler]] = Registry("LRScheduler", normalize_key=str.lower)


class _TorchNoOpLRScheduler(optim.lr_scheduler.LRScheduler):
    """PyTorch scheduler implementation that leaves parameter-group rates unchanged.

    Args:
        optimizer: Optimizer whose parameter-group learning rates remain unchanged.
        last_epoch: Index of the last epoch when resuming; defaults to ``-1``.
    """

    def get_lr(self) -> list[float]:
        """Return the current learning rates unchanged.

        Returns:
            Current learning rate for each optimizer parameter group.
        """
        return [group["lr"] for group in self.optimizer.param_groups]


@LRSchedulerRegistry.register("step")
class StepLRScheduler(optim.lr_scheduler.StepLR, Scheduler):
    """Step learning-rate scheduler registered with XANESNET.

    Args:
        lr_scheduler_type: Registry key identifying this scheduler type; must be ``"step"``.
        optimizer: Optimizer whose learning rate is scheduled.
        step_size: Number of epochs between learning-rate decays.
        gamma: Multiplicative factor applied at each decay; defaults to ``0.1``.
        last_epoch: Index of the last epoch when resuming; defaults to ``-1``.
    """

    def __init__(
        self,
        lr_scheduler_type: str,
        optimizer: Optimizer,
        step_size: int,
        gamma: float,
        last_epoch: int,
    ) -> None:
        """Initialize ``StepLRScheduler``."""
        self._set_scheduler_type(lr_scheduler_type)
        super().__init__(
            optimizer=optimizer.torch_optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
        )


@LRSchedulerRegistry.register("multistep")
class MultiStepLRScheduler(optim.lr_scheduler.MultiStepLR, Scheduler):
    """Multi-step learning-rate scheduler registered with XANESNET.

    Args:
        lr_scheduler_type: Registry key identifying this scheduler type; must be ``"multistep"``.
        optimizer: Optimizer whose learning rate is scheduled.
        milestones: Epoch indices at which the learning rate decays.
        gamma: Multiplicative factor applied at each milestone; defaults to ``0.1``.
        last_epoch: Index of the last epoch when resuming; defaults to ``-1``.
    """

    def __init__(
        self,
        lr_scheduler_type: str,
        optimizer: Optimizer,
        milestones: list[int],
        gamma: float,
        last_epoch: int,
    ) -> None:
        """Initialize ``MultiStepLRScheduler``."""
        self._set_scheduler_type(lr_scheduler_type)
        super().__init__(
            optimizer=optimizer.torch_optimizer,
            milestones=milestones,
            gamma=gamma,
            last_epoch=last_epoch,
        )


@LRSchedulerRegistry.register("exponential")
class ExponentialLRScheduler(optim.lr_scheduler.ExponentialLR, Scheduler):
    """Exponential learning-rate scheduler registered with XANESNET.

    Args:
        lr_scheduler_type: Registry key identifying this scheduler type; must be ``"exponential"``.
        optimizer: Optimizer whose learning rate is scheduled.
        gamma: Multiplicative factor applied at each epoch.
        last_epoch: Index of the last epoch when resuming; defaults to ``-1``.
    """

    def __init__(
        self,
        lr_scheduler_type: str,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int,
    ) -> None:
        """Initialize ``ExponentialLRScheduler``."""
        self._set_scheduler_type(lr_scheduler_type)
        super().__init__(
            optimizer=optimizer.torch_optimizer,
            gamma=gamma,
            last_epoch=last_epoch,
        )


@LRSchedulerRegistry.register("linear")
class LinearLRScheduler(optim.lr_scheduler.LinearLR, Scheduler):
    """Linear learning-rate scheduler registered with XANESNET.

    Args:
        lr_scheduler_type: Registry key identifying this scheduler type; must be ``"linear"``.
        optimizer: Optimizer whose learning rate is scheduled.
        start_factor: Initial learning-rate multiplier; defaults to ``1/3``.
        end_factor: Final learning-rate multiplier; defaults to ``1.0``.
        total_iters: Number of iterations over which the multiplier changes; defaults to ``5``.
        last_epoch: Index of the last epoch when resuming; defaults to ``-1``.
    """

    def __init__(
        self,
        lr_scheduler_type: str,
        optimizer: Optimizer,
        start_factor: float,
        end_factor: float,
        total_iters: int,
        last_epoch: int,
    ) -> None:
        """Initialize ``LinearLRScheduler``."""
        self._set_scheduler_type(lr_scheduler_type)
        super().__init__(
            optimizer=optimizer.torch_optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
            last_epoch=last_epoch,
        )


@LRSchedulerRegistry.register("constant")
class ConstantLRScheduler(optim.lr_scheduler.ConstantLR, Scheduler):
    """Constant learning-rate scheduler registered with XANESNET.

    Args:
        lr_scheduler_type: Registry key identifying this scheduler type; must be ``"constant"``.
        optimizer: Optimizer whose learning rate is scheduled.
        factor: Learning-rate multiplier applied during the constant phase; defaults to ``1/3``.
        total_iters: Number of iterations for which the multiplier is applied; defaults to ``5``.
        last_epoch: Index of the last epoch when resuming; defaults to ``-1``.
    """

    def __init__(
        self,
        lr_scheduler_type: str,
        optimizer: Optimizer,
        factor: float,
        total_iters: int,
        last_epoch: int,
    ) -> None:
        """Initialize ``ConstantLRScheduler``."""
        self._set_scheduler_type(lr_scheduler_type)
        super().__init__(
            optimizer=optimizer.torch_optimizer,
            factor=factor,
            total_iters=total_iters,
            last_epoch=last_epoch,
        )


@LRSchedulerRegistry.register("none")
@LRSchedulerRegistry.register("no")
class NoOpLRScheduler(_TorchNoOpLRScheduler, Scheduler):
    """No-op learning-rate scheduler registered with XANESNET.

    Args:
        lr_scheduler_type: Registry key identifying this scheduler type; must be ``"none"`` or ``"no"``.
        optimizer: Optimizer whose learning rate is left unchanged.
        last_epoch: Index of the last epoch when resuming; defaults to ``-1``.
    """

    def __init__(self, lr_scheduler_type: str, optimizer: Optimizer, last_epoch: int) -> None:
        """Initialize ``NoOpLRScheduler``."""
        self._set_scheduler_type(lr_scheduler_type)
        super().__init__(optimizer=optimizer.torch_optimizer, last_epoch=last_epoch)
