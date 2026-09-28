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

"""Base reporter interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from xanesnet.serialization.config import Config

from ..result import AnalysisResults


class Reporter(ABC):
    """Base class for analysis reporters.

    Reporters consume analysis results and write machine-readable files such as CSV, JSON, or YAML.

    Args:
        reporter_type: Registered reporter name from the analysis configuration.

    Attributes:
        reporter_type: Registered reporter name from the analysis configuration.
    """

    def __init__(self, reporter_type: str) -> None:
        """Initialize a reporter instance."""
        self.reporter_type = reporter_type

    @abstractmethod
    def report(
        self,
        results: AnalysisResults,
        output_dir: Path,
    ) -> None:
        """Generate report files from analysis results.

        Args:
            results: Analysis pipeline outputs to report.
            output_dir: Directory where report files should be written.
        """
        ...

    @property
    def signature(self) -> Config:
        """Return the reporter signature.

        Returns:
            Configuration values needed to recreate this reporter.
        """
        return Config({"reporter_type": self.reporter_type})

    def __str__(self) -> str:
        """Return the short display label of this reporter."""
        return self.reporter_type

    def __repr__(self) -> str:
        """Return a detailed representation of this reporter."""
        args = ", ".join(f"{key}={value!r}" for key, value in self.signature.as_dict().items())
        return f"{type(self).__name__}({args})"
