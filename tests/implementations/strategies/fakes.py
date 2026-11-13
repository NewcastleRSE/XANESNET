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

"""Lightweight stand-ins for the collaborators of the strategies.

They are registered under ``unit_test_*`` keys in the public registries; registry keys are permanent, so every fake
lives in this single module.
"""

import copy
from types import SimpleNamespace
from typing import Any

import torch

from xanesnet.models import ModelRegistry
from xanesnet.runners.inferencers import InferencerRegistry
from xanesnet.runners.trainers import TrainerRegistry
from xanesnet.serialization.config import Config

MODEL_TYPE = "unit_test_strategy_model"
TRAINER_TYPE = "unit_test_strategy_trainer"
INFERENCER_TYPE = "unit_test_strategy_inferencer"
N_ENERGIES = 6

EVENTS: list[tuple[str, Any]] = []
"""Chronological ``(event, payload)`` log shared by the fakes; cleared before every test."""


@ModelRegistry.register(MODEL_TYPE)
class FakeModel(torch.nn.Module):
    """Registered model stand-in that records initialization calls."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.linear = torch.nn.Linear(1, 1)
        self.init_calls: list[tuple[str, str, dict[str, Any]]] = []

    def init_weights(self, weights_init: str, bias_init: str, **kwargs: Any) -> None:
        self.init_calls.append((weights_init, bias_init, kwargs))

    @property
    def signature(self) -> Config:
        return Config({"model_type": MODEL_TYPE, **self.kwargs})


@TrainerRegistry.register(TRAINER_TYPE)
class FakeTrainer:
    """Registered trainer stand-in that records training calls."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model = kwargs["model"]
        self.dataset = kwargs["dataset"]
        EVENTS.append(("trainer_created", self))

    def train(self) -> None:
        EVENTS.append(("train", self))
        if getattr(self.model, "fail_on_train", False):
            raise RuntimeError("training failed")


@InferencerRegistry.register(INFERENCER_TYPE)
class FakeInferencer:
    """Registered inferencer stand-in that records inference calls."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def infer(self, predictions_save_path: Any) -> None:
        EVENTS.append(("infer", predictions_save_path))


class FakeDataset:
    """Minimal dataset with the split API the strategies rely on."""

    def __init__(self, size: int = 10, plans: list[list[list[int]]] | None = None, indexfile: bool = False) -> None:
        self.size = size
        self.configured_plans = plans if plans is not None else [[list(range(size - 2)), [size - 2, size - 1]]]
        self.has_split_indexfile = indexfile
        self._subsets: list[list[list[int]]] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(energies=torch.arange(N_ENERGIES, dtype=torch.float32))

    @property
    def signature(self) -> Config:
        return Config({"dataset_type": "fake"})

    def setup_splits(self) -> list[list[list[int]]]:
        self.set_splits(self.configured_plans)
        return self.configured_plans

    def set_splits(self, plans: list[list[list[int]]]) -> None:
        self._subsets = [[list(indices) for indices in plan] for plan in plans]

    def get_dataset(self, idx: int) -> "FakeDataset":
        view = copy.copy(self)
        view._subsets = [self._subsets[idx]]
        return view

    def get_subset_indices(self, index: int) -> list[int] | None:
        if not self._subsets or index >= len(self._subsets[0]):
            return None
        return list(self._subsets[0][index])

    @property
    def train_subset(self) -> list[int] | None:
        return self.get_subset_indices(0)

    @property
    def plans(self) -> list[list[list[int]]]:
        return self._subsets


def event_names() -> list[str]:
    """Return fake-collaborator event names in chronological order.

    Returns:
        Event names, including repeated events.
    """
    return [name for name, _ in EVENTS]


class FakeCheckpointer:
    """Records ``finish_member`` instead of collecting real optimizer/scheduler state."""

    def finish_member(self, trainer: Any) -> None:
        EVENTS.append(("finish_member", trainer))


ENCODING_CONFIG = Config({"model": {"model_type": MODEL_TYPE}, "encodings": [{"encoding_type": "none"}]})
