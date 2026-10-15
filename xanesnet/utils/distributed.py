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


import os
from pathlib import Path

_RUN_DIR = "XANESNET_RUN_DIR"


def is_ddp_child() -> bool:
    """Return whether the current process is a XANESNET DDP child."""
    value = os.environ.get("LOCAL_RANK")
    return value is not None and int(value) > 0


def is_local_rank_zero() -> bool:
    """Return whether the current process is local rank 0."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def set_ddp_run_dir(save_dir: Path) -> None:
    """Make the prepared run directory available to DDP child processes."""
    os.environ[_RUN_DIR] = str(save_dir)


def get_ddp_run_dir() -> Path | None:
    """Return the parent run directory when running as a DDP child."""
    value = os.environ.get(_RUN_DIR)
    return Path(value) if value else None
