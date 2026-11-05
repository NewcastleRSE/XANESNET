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


"""Lightweight environment context for prepared Lightning training runs."""

import os
from pathlib import Path

from xanesnet.utils.exceptions import ConfigError

LIGHTNING_CONTEXT_ENV = "XANESNET_LIGHTNING_CONTEXT"
LIGHTNING_LAUNCH_ENVIRONMENT = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT")


def get_lightning_context_path() -> Path | None:
    """Read the prepared-run context without importing Lightning.

    Returns:
        Absolute path when a Lightning context is present, otherwise ``None``.

    Raises:
        ConfigError: If the context is empty or is not an absolute path.
    """
    value = os.environ.get(LIGHTNING_CONTEXT_ENV)
    if value is None:
        return None
    path = Path(value)
    if not value or not path.is_absolute():
        raise ConfigError(f"{LIGHTNING_CONTEXT_ENV} must contain an absolute context path.")
    return path
