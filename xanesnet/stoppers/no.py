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

"""No-op early stopper that never halts training."""

from xanesnet.models import Model

from .base import EarlyStopper
from .registry import EarlyStopperRegistry


@EarlyStopperRegistry.register("none")
@EarlyStopperRegistry.register("no")
class NoStopper(EarlyStopper):
    """No-op early stopper that never halts training.

    Still tracks the best metric value and optionally snapshots model
    weights (when ``restore_best`` is ``True``) so that ``restore`` can
    be used after training completes.

    Args:
        early_stopper_type: Registry key identifying this stopper type.
        restore_best: Passed through to ``EarlyStopper``; controls
            whether model weights are snapshotted on improvement.
    """

    def __init__(
        self,
        early_stopper_type: str,
        restore_best: bool,
    ) -> None:
        """Initialize the no-op stopper."""
        super().__init__(early_stopper_type, restore_best)

    def step(self, value: float, model: Model, epoch: int) -> bool:
        """Update tracking state and always return ``False``.

        Args:
            value: Current metric value to minimize.
            model: The model being trained.
            epoch: Current epoch index.

        Returns:
            Always ``False``.
        """
        _ = super().step(value, model, epoch)
        return False
