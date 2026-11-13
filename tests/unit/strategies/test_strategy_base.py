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

"""Unit tests for behavior shared by all ``Strategy`` implementations."""

import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import xanesnet.strategies.base as strategy_base
from xanesnet.models import Model
from xanesnet.serialization.config import Config
from xanesnet.strategies.base import Strategy
from xanesnet.utils.exceptions import ConfigError


class MinimalDataset:
    """Dataset stub with configurable split plans and a spectral energy grid."""

    def __init__(self, plans=None, energies=torch.tensor([1.0, 2.0, 3.0])) -> None:
        self.plans = plans if plans is not None else [[[0, 1, 2, 3], [4, 5]]]
        self.energies = energies
        self.train_subset = None
        self.signature = Config({"dataset_type": "minimal"})

    def setup_splits(self):
        self.train_subset = object() if self.plans else None
        return self.plans

    def get_subset_indices(self, subset_index):
        if not self.plans or subset_index >= len(self.plans[0]):
            return None
        return self.plans[0][subset_index]

    def __getitem__(self, index):
        return SimpleNamespace(energies=self.energies)


class MinimalModel(Model):
    """Model stub that records the shared strategy's weight initialization."""

    def __init__(self, model_config: Config) -> None:
        super().__init__("minimal")
        self.linear = torch.nn.Linear(1, 1)
        self._signature = model_config
        self.init_calls: list[tuple[str, str, dict[str, Any]]] = []

    def init_weights(self, weights_init: str, bias_init: str, **kwargs: Any) -> None:
        self.init_calls.append((weights_init, bias_init, kwargs))

    @property
    def signature(self) -> Config:
        return self._signature


class MinimalStrategy(Strategy):
    """Concrete strategy stub used to exercise inherited base behavior."""

    def __init__(self, model_count: int = 1, **kwargs: Any) -> None:
        self.model_count = model_count
        super().__init__(**kwargs)

    def setup_models(self, model_config: Config) -> None:
        self.models.clear()
        for _ in range(self.model_count):
            self.models.append(MinimalModel(model_config))

    def setup_trainers(self, device: str | torch.device) -> None:
        self.trainers = [None] * self.model_count

    def run_training(self) -> list[Model]:
        return super().run_training()

    def setup_inferencers(self, device: str | torch.device) -> None:
        self.inferencer = None

    def run_inference(self, predictions_save_path: str | Path | None) -> None:
        super().run_inference(predictions_save_path)


class MinimalEncoding:
    """Encoding stub that records the energy-grid length used during setup."""

    def __init__(self) -> None:
        self.prepared_for = None

    def prepare(self, n_energies: int) -> None:
        self.prepared_for = n_energies

    def output_size(self, n_energies: int) -> int:
        return n_energies

    @property
    def signature(self) -> list[Config]:
        return [Config({"encoding_type": "minimal"})]


def install_minimal_encoding(monkeypatch: pytest.MonkeyPatch) -> MinimalEncoding:
    """Replace encoding resolution and construction with a local test stub.

    Args:
        monkeypatch: Pytest fixture used to replace strategy dependencies.

    Returns:
        The encoding stub returned by the patched factory.
    """
    encoding = MinimalEncoding()
    monkeypatch.setattr(strategy_base, "resolve_auto_encoding_config", lambda config, dataset: config)
    monkeypatch.setattr(
        strategy_base,
        "CombinedEncoding",
        SimpleNamespace(from_configs=lambda configs: encoding),
    )
    return encoding


@pytest.fixture
def build() -> Callable[..., MinimalStrategy]:
    """Provide a factory for constructing minimal strategies.

    Returns:
        Factory accepting a dataset, model count, and strategy overrides.
    """

    def factory(dataset=None, model_count: int = 1, **overrides: Any) -> MinimalStrategy:
        """Construct a strategy with resolved test configuration values."""
        kwargs = {
            "strategy_type": "minimal",
            "dataset": dataset if dataset is not None else MinimalDataset(),
            "weight_init": "xavier",
            "weight_init_params": Config({"gain": 2.0}),
            "bias_init": "zeros",
            "checkpoint_dir": None,
            "checkpoint_interval": None,
            "tensorboard_dir": None,
            "trainer_config": Config({"trainer_type": "minimal"}),
            "inferencer_config": Config({"inferencer_type": "minimal"}),
        }
        kwargs.update(overrides)
        return MinimalStrategy(model_count=model_count, **kwargs)

    return factory


@pytest.fixture
def model_config() -> Config:
    """Provide the minimal model configuration used by strategy tests.

    Returns:
        Configuration identifying the test model and its width.
    """
    return Config({"model_type": "minimal", "width": 3})


def test_a_strategy_needs_a_trainer_or_an_inferencer_config(build):
    with pytest.raises(ValueError, match="Either trainer_config or inferencer_config"):
        build(trainer_config=None, inferencer_config=None)

    assert build(trainer_config=None).trainer_config is None
    assert build(inferencer_config=None).inferencer_config is None


def test_weights_are_initialised_with_the_configured_schemes_on_every_model(build, model_config):
    strategy = build(model_count=2)
    strategy.setup_models(model_config)

    strategy.init_model_weights()

    assert [model.init_calls for model in strategy.models] == [[("xavier", "zeros", {"gain": 2.0})]] * 2


def test_weights_cannot_be_initialised_before_the_models_exist(build):
    with pytest.raises(ValueError, match="models are not initialized"):
        build().init_model_weights()


def test_state_dicts_are_loaded_into_the_models_in_order(build, model_config):
    strategy = build(model_count=2)
    strategy.setup_models(model_config)
    states = [
        {"linear.weight": torch.full((1, 1), 1.0), "linear.bias": torch.zeros(1)},
        {"linear.weight": torch.full((1, 1), 2.0), "linear.bias": torch.zeros(1)},
    ]

    strategy.set_state_dicts(states)

    assert [model.linear.weight.item() for model in strategy.models] == [1.0, 2.0]


def test_too_few_checkpoint_states_are_an_error(build, model_config):
    strategy = build(model_count=2)
    strategy.setup_models(model_config)

    with pytest.raises(ConfigError, match="1 model states; strategy requires 2"):
        strategy.set_state_dicts([strategy.models[0].state_dict()])


def test_extra_checkpoint_states_are_ignored_with_a_warning(build, model_config, caplog):
    strategy = build()
    strategy.setup_models(model_config)
    first = {"linear.weight": torch.full((1, 1), 5.0), "linear.bias": torch.zeros(1)}
    extra = {"linear.weight": torch.full((1, 1), 9.0), "linear.bias": torch.zeros(1)}

    with caplog.at_level(logging.WARNING):
        strategy.set_state_dicts([first, extra])

    assert strategy.models[0].linear.weight.item() == 5.0
    assert "1 extra model states" in caplog.text


def test_state_dicts_cannot_be_loaded_before_the_models_exist(build):
    with pytest.raises(ValueError, match="models are not initialized"):
        build().set_state_dicts([{}])


def test_splits_come_from_the_dataset(build):
    dataset = MinimalDataset(plans=[[[0, 1, 2, 3], [4, 5]]])

    assert build(dataset).setup_model_splits() == [[[0, 1, 2, 3], [4, 5]]]
    assert dataset.get_subset_indices(1) == [4, 5]


def test_the_encoding_is_built_from_config_and_prepared_for_the_energy_grid(build, monkeypatch):
    strategy = build()
    strategy.setup_model_splits()
    encoding = install_minimal_encoding(monkeypatch)

    configured = strategy.setup_encoding(Config({"encodings": [{"encoding_type": "minimal"}]}))

    assert strategy.encoding is configured is encoding
    assert encoding.output_size(3) == 3
    assert encoding.prepared_for == 3


def test_training_requires_splits_before_encoding(build, monkeypatch):
    strategy = build(dataset=MinimalDataset(plans=[]))
    install_minimal_encoding(monkeypatch)

    with pytest.raises(ValueError, match="before model splits are initialized"):
        strategy.setup_encoding(Config({"encodings": []}))


def test_inference_can_build_encoding_without_splits(build, monkeypatch):
    strategy = build(dataset=MinimalDataset(plans=[]), trainer_config=None)
    encoding = install_minimal_encoding(monkeypatch)

    assert strategy.setup_encoding(Config({"encodings": []})) is encoding


@pytest.mark.parametrize("energies", [None, torch.tensor(1.0)])
def test_samples_without_an_energy_grid_are_rejected(build, monkeypatch, energies):
    strategy = build(dataset=MinimalDataset(energies=energies))
    strategy.setup_model_splits()
    install_minimal_encoding(monkeypatch)

    with pytest.raises(ValueError, match="spectral energy grid"):
        strategy.setup_encoding(Config({"encodings": []}))


def test_checkpointer_signature_combines_dataset_model_and_encoding(build, model_config, tmp_path):
    strategy = build(checkpoint_dir=tmp_path / "ckpt", checkpoint_interval=5)
    strategy.encoding = MinimalEncoding()
    strategy.setup_models(model_config)

    checkpointer = strategy.setup_checkpointer()

    assert strategy.checkpointer is checkpointer
    assert checkpointer.save_dir == tmp_path / "ckpt"
    assert checkpointer.save_interval == 5
    assert checkpointer.signature.as_dict() == {
        "dataset": {"dataset_type": "minimal"},
        "model": {"model_type": "minimal", "width": 3},
        "encodings": [{"encoding_type": "minimal"}],
    }


@pytest.mark.parametrize(
    ("with_encoding", "message"),
    [(False, "encoding is not initialized"), (True, "Models are not initialized")],
)
def test_checkpointer_needs_encoding_and_models(build, with_encoding, message):
    strategy = build()
    if with_encoding:
        strategy.encoding = MinimalEncoding()

    with pytest.raises(ValueError, match=message):
        strategy.setup_checkpointer()


def test_checkpointer_is_only_for_training(build, model_config):
    strategy = build(trainer_config=None)
    strategy.encoding = MinimalEncoding()
    strategy.setup_models(model_config)

    with pytest.raises(ValueError, match="no trainer config"):
        strategy.setup_checkpointer()


def test_the_checkpointer_sees_the_current_trainers(build, model_config):
    strategy = build()
    strategy.encoding = MinimalEncoding()
    strategy.setup_models(model_config)
    strategy.setup_checkpointer()
    strategy.setup_trainers("cpu")

    assert strategy.checkpointer.trainer_provider() is strategy.trainers


def test_model_signature_requires_models(build):
    with pytest.raises(ValueError, match="Models are not initialized"):
        _ = build().model_signature


def test_model_signature_is_that_of_the_first_model(build, model_config):
    strategy = build(model_count=2)
    strategy.setup_models(model_config)

    assert strategy.model_signature.as_dict() == {"model_type": "minimal", "width": 3}
