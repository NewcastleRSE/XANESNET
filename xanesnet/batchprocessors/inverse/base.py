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

"""Direction-specific base class for inverse XANESNET batch processors."""

from typing import Any

import torch

from ..base import BatchProcessor


class InverseBatchProcessor(BatchProcessor):
    """Base class for inverse batch processors (spectra are the model input).

    Inverse prediction maps a spectrum to a structure representation
    (descriptors, properties, ...). The spectral encoding is therefore applied
    to the input, while targets and predictions -- structural quantities --
    pass through unchanged. Concrete inverse processors only implement the
    data-shaping methods and may override :attr:`spectra_input_key` when the
    spectral input is not stored under ``"x"``.
    """

    spectra_input_key: str = "x"

    def encode_input(self, inputs: dict[str, Any], elements: torch.Tensor | None = None) -> dict[str, Any]:
        """Encode the spectral input under :attr:`spectra_input_key`.

        Args:
            inputs: Input dict returned by :meth:`input_preparation`.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.

        Returns:
            New input dict with the spectral entry encoded when an encoding is
            configured, otherwise ``inputs`` unchanged.
        """
        if self._encoding is None:
            return inputs
        encoded = dict(inputs)
        encoded[self.spectra_input_key] = self._encoding.encode(inputs[self.spectra_input_key], elements)
        return encoded

    def encode_target(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Return targets unchanged; inverse targets are structural, not spectra.

        Args:
            targets: Raw structural target tensor.
            elements: Unused; present for interface compatibility.

        Returns:
            ``targets`` unchanged.
        """
        return targets

    def decode_target(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Return predictions unchanged; inverse predictions are structural.

        Args:
            predictions: Model output tensor (structural quantities).
            elements: Unused; present for interface compatibility.

        Returns:
            ``predictions`` unchanged.
        """
        return predictions
