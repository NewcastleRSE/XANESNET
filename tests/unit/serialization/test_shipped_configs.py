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

"""Guard against schema drift: every shipped example configuration must still validate."""

import copy
from pathlib import Path

import pytest

from xanesnet.serialization.config import load_raw_config
from xanesnet.serialization.schema_validation import validate_config_schema

from ..conftest import CONFIGS_DIR, DRY_RUNS_DIR


def _yaml_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.yaml")) if directory.is_dir() else []


def _relative(path: Path) -> str:
    return path.relative_to(CONFIGS_DIR.parent).as_posix()


# ``*_infer.yaml`` files are *user* inference configs: they only validate once merged with a checkpoint signature.
TRAIN_CONFIGS = [p for p in _yaml_files(CONFIGS_DIR) if not p.stem.endswith("_infer") and "analyze" not in p.stem] + (
    _yaml_files(DRY_RUNS_DIR / "train")
)
ANALYZE_CONFIGS = [p for p in _yaml_files(CONFIGS_DIR) if "analyze" in p.stem] + _yaml_files(DRY_RUNS_DIR / "analyze")


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=_relative)
def test_training_configs_validate(path):
    validated = validate_config_schema(load_raw_config(path), "train")
    assert "trainer" in validated


@pytest.mark.parametrize("path", ANALYZE_CONFIGS, ids=_relative)
def test_analysis_configs_validate(path):
    validated = validate_config_schema(load_raw_config(path), "analyze")
    assert "collectors" in validated


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=_relative)
def test_validated_training_configs_are_a_fixed_point(path):
    """Re-validating a validated config changes nothing, so saved ``validated_*`` files can be reused."""
    once = validate_config_schema(load_raw_config(path), "train")
    assert validate_config_schema(copy.deepcopy(once), "train") == once


def test_example_configs_were_discovered():
    assert TRAIN_CONFIGS and ANALYZE_CONFIGS
