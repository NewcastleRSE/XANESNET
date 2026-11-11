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

"""Shared fixtures for the unit tests: minimal configs, helpers and global-state isolation."""

import copy
import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from xanesnet.serialization._types import ConfigRaw
from xanesnet.utils.lightning import LIGHTNING_CONTEXT_ENV
from xanesnet.utils.prompts import is_auto_yes_enabled, set_auto_yes

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DRY_RUNS_DIR = REPO_ROOT / "tests" / "dry_runs"


###############################################################################
############################## CONFIG FACTORIES ###############################
###############################################################################


def make_train_config() -> ConfigRaw:
    """MLP on pmgjson data config"""
    return {
        "datasource": {
            "datasource_type": "pmgjson",
            "json_path": "./data/toy_data/",
            "spectrum_key": "XANES",
        },
        "dataset": {
            "dataset_type": "descriptor",
            "root": "./data/processed/unit_test/",
            "descriptors": [
                {"descriptor_type": "wacsf", "r_min": 1.0, "r_max": 6.0, "n_g2": 2, "n_g4": 2},
            ],
        },
        "encodings": [{"encoding_type": "identity"}],
        "model": {
            "model_type": "mlp",
            "in_size": "auto",
            "out_size": "auto",
            "hidden_size": 8,
            "num_hidden_layers": 1,
        },
        "trainer": {
            "trainer_type": "basic",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss": [{"loss_type": "mse"}],
        },
        "strategy": {"strategy_type": "single"},
    }


def make_infer_user_config() -> ConfigRaw:
    """Minimal inference config"""
    return {
        "datasource": {
            "datasource_type": "pmgjson",
            "json_path": "./data/toy_data/",
            "spectrum_key": "XANES",
        },
        "dataset": {"root": "./data/processed/unit_test/"},
        "strategy": {"strategy_type": "single"},
        "inferencer": {"inferencer_type": "basic", "batch_size": 2},
    }


def make_signature() -> ConfigRaw:
    """Checkpoint signature matching train config"""
    return {
        "dataset": {
            "dataset_type": "descriptor",
            "descriptors": [
                {"descriptor_type": "wacsf", "r_min": 1.0, "r_max": 6.0, "n_g2": 2, "n_g4": 2},
            ],
        },
        "model": {
            "model_type": "mlp",
            "in_size": 6,
            "out_size": 10,
            "hidden_size": 8,
            "num_hidden_layers": 1,
        },
        "encodings": [{"encoding_type": "identity"}],
    }


def make_analyze_config() -> ConfigRaw:
    """Minimal analysis config."""
    return {
        "selectors": [{"selector_type": "all"}],
        "collectors": [{"collector_type": "loss", "loss": {"loss_type": "mse"}}],
    }


###############################################################################
################################### FIXTURES ##################################
###############################################################################


@pytest.fixture(autouse=True)
def _isolate_global_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo process-wide side effects (logging, std streams, auto-yes) after every test.

    The XANESNET entry points configure the root logger, replace ``sys.stdout``/``sys.stderr`` and toggle global
    prompt state. Restoring them keeps tests order-independent.
    """
    monkeypatch.delenv(LIGHTNING_CONTEXT_ENV, raising=False)

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    streams = sys.stdout, sys.stderr
    auto_yes_enabled = is_auto_yes_enabled()

    yield

    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)
    sys.stdout, sys.stderr = streams
    set_auto_yes(auto_yes_enabled)


@pytest.fixture
def train_config() -> ConfigRaw:
    """Fresh minimal training config."""
    return make_train_config()


@pytest.fixture
def infer_user_config() -> ConfigRaw:
    """Fresh minimal user inference config."""
    return make_infer_user_config()


@pytest.fixture
def signature() -> ConfigRaw:
    """Fresh minimal checkpoint signature."""
    return make_signature()


@pytest.fixture
def analyze_config() -> ConfigRaw:
    """Fresh minimal analysis config."""
    return make_analyze_config()


@pytest.fixture
def mutate() -> Callable[[ConfigRaw, Callable[[ConfigRaw], Any]], ConfigRaw]:
    """Return a helper that applies ``fn`` to a deep copy of a config and returns the copy."""

    def _mutate(base: ConfigRaw, fn: Callable[[ConfigRaw], Any]) -> ConfigRaw:
        config = copy.deepcopy(base)
        fn(config)
        return config

    return _mutate


@pytest.fixture
def write_yaml(tmp_path: Path) -> Callable[..., Path]:
    """Return a helper that writes a YAML document (dict or raw text) into ``tmp_path``."""

    def _write(content: Any, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        text = content if isinstance(content, str) else yaml.safe_dump(content, sort_keys=False)
        path.write_text(text)
        return path

    return _write
