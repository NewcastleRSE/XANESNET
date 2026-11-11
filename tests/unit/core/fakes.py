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

"""Lightweight test doubles for the core pipelines.

The fakes are registered in the public data source, dataset and strategy registries under ``unit_test_*`` keys, so the
pipelines resolve them exactly like real implementations. Everything they do is recorded in ``EVENTS``.
"""

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from xanesnet.datasets import DatasetRegistry
from xanesnet.datasources import DataSourceRegistry
from xanesnet.serialization.config import Config
from xanesnet.strategies import StrategyRegistry

DATASOURCE_TYPE = "unit_test_datasource"
DATASET_TYPE = "unit_test_dataset_core"
STRATEGY_TYPE = "unit_test_strategy"

EVENTS: list[tuple[str, Any]] = []
"""Chronological ``(event, payload)`` log shared by all fakes; cleared before every test by ``events``."""

SPLITS = [[[0, 1, 2], [3]]]
SIGNATURE = {"model": {"model_type": "fake"}, "encodings": [], "dataset": {"dataset_type": DATASET_TYPE}}


class FakeModel:
    def __init__(self) -> None:
        self.device: torch.device | None = None

    def to(self, device: torch.device) -> "FakeModel":
        self.device = device
        return self


class Recorder:
    """Object whose methods/attributes record calls; used as optimizer/scheduler stand-in."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.loaded: list[Any] = []

    def load_state_dict(self, state: Any) -> None:
        self.loaded.append(state)
        EVENTS.append((f"load_{self.name}", state))


class FakeCheckpointer:
    signature = Config(SIGNATURE)

    def save(self, path: Path) -> None:
        EVENTS.append(("checkpointer_save", Path(path)))


class FakeEncoding:
    signature = [Config({"encoding_type": "fake_encoding", "num_points": 3})]


@DataSourceRegistry.register(DATASOURCE_TYPE)
class FakeDataSource:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        EVENTS.append(("datasource", kwargs))


@DatasetRegistry.register(DATASET_TYPE)
class FakeDataset:
    def __init__(self, datasource: FakeDataSource, **kwargs: Any) -> None:
        self.datasource = datasource
        self.kwargs = kwargs
        EVENTS.append(("dataset", kwargs))

    def prepare(self) -> None:
        EVENTS.append(("prepare", None))

    def check_preload(self) -> None:
        EVENTS.append(("check_preload", None))

    def __len__(self) -> int:
        return 4


@StrategyRegistry.register(STRATEGY_TYPE)
class FakeStrategy:
    """Strategy double with one trainer whose optimizer/schedulers are recorders."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.models = [FakeModel()]
        self.trainers = [
            SimpleNamespace(
                optimizer=SimpleNamespace(torch_optimizer=Recorder("optimizer")),
                epoch_lr_scheduler=Recorder("epoch_scheduler"),
                warmup_lr_scheduler=Recorder("warmup_scheduler"),
            )
        ]
        EVENTS.append(("strategy", kwargs))

    def setup_model_splits(self) -> list[list[list[int]]]:
        EVENTS.append(("setup_model_splits", None))
        return SPLITS

    def setup_encoding(self, config: Config) -> FakeEncoding:
        EVENTS.append(("setup_encoding", config))
        return FakeEncoding()

    def setup_models(self, model_config: Config) -> None:
        EVENTS.append(("setup_models", model_config))

    def setup_checkpointer(self) -> FakeCheckpointer:
        EVENTS.append(("setup_checkpointer", None))
        return FakeCheckpointer()

    def init_model_weights(self) -> None:
        EVENTS.append(("init_model_weights", None))

    def setup_trainers(self, device: str) -> None:
        EVENTS.append(("setup_trainers", device))

    def setup_inferencers(self, device: str) -> None:
        EVENTS.append(("setup_inferencers", device))

    def set_state_dicts(self, states: list[Mapping[str, Any]]) -> None:
        EVENTS.append(("set_state_dicts", states))

    def run_training(self) -> list[FakeModel]:
        EVENTS.append(("run_training", None))
        return self.models

    def run_inference(self, predictions_path: Path | None) -> None:
        EVENTS.append(("run_inference", predictions_path))


def event_names() -> list[str]:
    return [name for name, _ in EVENTS]


def payload(name: str) -> Any:
    """Payload of the single event called ``name``."""
    matches = [value for event, value in EVENTS if event == name]
    assert len(matches) == 1, f"expected exactly one {name!r} event, got {len(matches)}"
    return matches[0]
