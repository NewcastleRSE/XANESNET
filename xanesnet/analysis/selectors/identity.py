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

"""Selector that yields each prediction sample unchanged."""

from collections.abc import Iterator

from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample

from .base import Selector
from .registry import SelectorRegistry


@SelectorRegistry.register("all")
@SelectorRegistry.register("none")
class IdentitySelector(Selector):
    """Yield all source samples unchanged.

    The registered ``all`` and ``none`` selector names both use this identity behavior.

    Args:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader to select samples from.
    """

    def __init__(self, selector_type: str, data_source: PredictionReader) -> None:
        """Initialize an identity selector."""
        super().__init__(selector_type, data_source)

    def __iter__(self) -> Iterator[PredictionSample]:
        """Yield all samples from ``data_source``.

        Returns:
            Iterator over prediction samples.
        """
        yield from self.data_source

    def __len__(self) -> int:
        """Return the number of selected samples.

        Returns:
            Number of selected prediction samples.
        """
        return len(self.data_source)
