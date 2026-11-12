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

"""Fixtures shared by the strategy tests."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from xanesnet.runners.inferencers import InferencerRegistry
from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import tb_logger
from xanesnet.strategies import StrategyRegistry

from . import fakes


@pytest.fixture(autouse=True)
def events() -> Iterator[list]:
    fakes.EVENTS.clear()
    yield fakes.EVENTS
    fakes.EVENTS.clear()


@pytest.fixture(autouse=True)
def tensorboard_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """Record ``tb_logger`` usage instead of writing event files."""
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(tb_logger, "new_run", lambda path: calls.append(("new_run", Path(path))))
    monkeypatch.setattr(tb_logger, "close", lambda: calls.append(("close", None)))
    return calls


@pytest.fixture
def ensemble_inferencers(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Replace the registry factory so the hard-wired ``ensemble`` inferencer becomes a recording fake."""
    created: list[tuple[str, dict[str, Any]]] = []
    original = InferencerRegistry.create

    def create(key: str, **kwargs: Any) -> Any:
        created.append((key, kwargs))
        return original(fakes.INFERENCER_TYPE, **kwargs)

    monkeypatch.setattr(InferencerRegistry, "create", create)
    return created


@pytest.fixture
def model_config() -> Config:
    return Config({"model_type": fakes.MODEL_TYPE, "width": 3})


@pytest.fixture
def trainer_config() -> Config:
    return Config({"trainer_type": fakes.TRAINER_TYPE, "epochs": 2})


@pytest.fixture
def inferencer_config() -> Config:
    return Config({"inferencer_type": fakes.INFERENCER_TYPE, "batch_size": 4})


@pytest.fixture
def build(model_config: Config, trainer_config: Config, inferencer_config: Config):
    """Create a strategy through the registry with fake collaborators."""

    def factory(key: str, dataset: fakes.FakeDataset | None = None, **overrides: Any):
        kwargs: dict[str, Any] = {
            "strategy_type": key,
            "dataset": dataset if dataset is not None else fakes.FakeDataset(),
            "weight_init": "xavier",
            "weight_init_params": Config({"gain": 2.0}),
            "bias_init": "zeros",
            "checkpoint_dir": None,
            "checkpoint_interval": None,
            "tensorboard_dir": None,
            "trainer_config": trainer_config,
            "inferencer_config": inferencer_config,
        }
        kwargs.update(overrides)
        return StrategyRegistry.create(key, **kwargs)

    return factory


@pytest.fixture
def prepared(build, model_config: Config):
    """Create a strategy and run the training set-up steps up to (excluding) the trainers."""

    def factory(key: str, **kwargs: Any):
        strategy = build(key, **kwargs)
        strategy.setup_model_splits()
        strategy.setup_encoding(fakes.ENCODING_CONFIG)
        strategy.setup_models(model_config)
        strategy.setup_checkpointer()
        strategy.checkpointer = fakes.FakeCheckpointer()
        return strategy

    return factory
