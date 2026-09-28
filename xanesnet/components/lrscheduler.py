# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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

"""Registry instance for learning-rate scheduler classes."""

import torch.optim as optim

from xanesnet.utils.registry import Registry

LRSchedulerRegistry: Registry[type[optim.lr_scheduler.LRScheduler]] = Registry(
    "LRScheduler",
    normalize_key=str.lower,
)


class NoOpLRScheduler(optim.lr_scheduler.LRScheduler):
    """Learning rate scheduler that leaves all parameter group learning rates unchanged.

    Args:
        optimizer: Wrapped optimizer whose learning rates are reported unchanged.
        last_epoch: Last epoch index passed to :class:`torch.optim.lr_scheduler.LRScheduler`.
    """

    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        """Initialize ``NoOpLRScheduler``."""
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        """Return the current learning rates unchanged.

        Returns:
            Current learning rate from each optimizer parameter group.
        """
        return [group["lr"] for group in self.optimizer.param_groups]


# register lrschedulers
LRSchedulerRegistry.register("step")(optim.lr_scheduler.StepLR)
LRSchedulerRegistry.register("multistep")(optim.lr_scheduler.MultiStepLR)
LRSchedulerRegistry.register("exponential")(optim.lr_scheduler.ExponentialLR)
LRSchedulerRegistry.register("linear")(optim.lr_scheduler.LinearLR)
LRSchedulerRegistry.register("constant")(optim.lr_scheduler.ConstantLR)
LRSchedulerRegistry.register("none")(NoOpLRScheduler)
LRSchedulerRegistry.register("no")(NoOpLRScheduler)
