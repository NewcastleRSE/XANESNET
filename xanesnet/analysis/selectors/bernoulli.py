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

"""Selector that samples fixed prediction indices with Bernoulli trials."""

import random
from collections.abc import Iterator

from xanesnet.serialization.config import Config
from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample

from .base import Selector
from .registry import SelectorRegistry


@SelectorRegistry.register("random")
class BernoulliSelector(Selector):
    """Select a random fixed subset of samples using Bernoulli sampling.

    Args:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader to select samples from.
        p: Probability of retaining each sample. Must be in the inclusive range ``[0, 1]``.
    """

    def __init__(self, selector_type: str, data_source: PredictionReader, p: float) -> None:
        """Initialize the selector and draw the retained sample indices."""
        super().__init__(selector_type, data_source)

        self.p = p
        self._selected_indices: list[int] = [i for i in range(len(data_source)) if random.random() < p]

    def __iter__(self) -> Iterator[PredictionSample]:
        """Yield the prediction samples retained at initialization time.

        Returns:
            Iterator over selected prediction samples.
        """
        for i in self._selected_indices:
            yield self.data_source[i]

    def __len__(self) -> int:
        """Return the number of selected samples.

        Returns:
            Number of selected prediction samples.
        """
        return len(self._selected_indices)

    @property
    def signature(self) -> Config:
        """Return the selector signature.

        Returns:
            Configuration values needed to recreate this selector.
        """
        signature = super().signature
        signature.update_with_dict({"p": self.p})
        return signature
