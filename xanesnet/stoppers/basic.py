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

"""Patience-based early stopper for XANESNET training."""

from xanesnet.models import Model

from .base import EarlyStopper
from .registry import EarlyStopperRegistry


@EarlyStopperRegistry.register("basic")
class BasicStopper(EarlyStopper):
    """Patience-based early stopper.

    Stops training when no meaningful improvement has been observed for
    ``patience`` consecutive epochs. Only meaningful improvements update the
    tracked best value and the optional restorable model snapshot.

    Args:
        early_stopper_type: Registry key identifying this stopper type.
        restore_best: If ``True``, restore the model to its best-seen
            state when ``restore`` is called.
        patience: Number of epochs without meaningful improvement after
            which training is stopped.
        min_delta: Minimum absolute improvement per epoch required to be
            counted as meaningful progress. The threshold is scaled by the
            number of epochs elapsed since the previous best.
    """

    def __init__(
        self,
        early_stopper_type: str,
        restore_best: bool,
        patience: int,
        min_delta: float = 0.0,
    ) -> None:
        """Initialize the patience-based stopper."""
        super().__init__(early_stopper_type, restore_best)

        self.patience = patience
        self.min_delta = min_delta
        self.last_improvement_epoch: int = 0

    def step(self, value: float, model: Model, epoch: int) -> bool:
        """Check whether training should stop.

        The first call establishes the initial best value. Later calls only
        count as improvements when they beat the current best by at least
        ``min_delta * epochs_since_best``.

        Args:
            value: Current metric value to minimize.
            model: The model being trained.
            epoch: Current epoch index.

        Returns:
            ``True`` if the number of epochs without meaningful improvement
            has reached ``patience``, ``False`` otherwise.
        """
        if self.best_epoch < 0:
            _ = super().step(value, model, epoch)
            self.last_improvement_epoch = epoch
            return False

        epochs_since_best = max(1, epoch - self.best_epoch)
        required_total_delta = self.min_delta * epochs_since_best
        meaningful_improvement = value < self.best_value - required_total_delta

        if meaningful_improvement:
            _ = super().step(value, model, epoch)
            self.last_improvement_epoch = epoch

        last_improvement = self.last_improvement_epoch
        epochs_without_improvement = epoch - last_improvement

        return epochs_without_improvement >= self.patience
