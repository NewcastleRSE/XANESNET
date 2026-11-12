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

"""Unit tests for the early stoppers (``basic``, ``time``, ``none``) and best-state restoring."""

from types import SimpleNamespace

import pytest
import torch

import xanesnet.stoppers.time as time_stopper
from xanesnet.components import LRSchedulerRegistry, OptimizerRegistry
from xanesnet.stoppers import EarlyStopper, EarlyStopperRegistry


class Training:
    """A tiny model with optimizer and schedulers, as handed to ``EarlyStopper.step``."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = OptimizerRegistry.create(
            "sgd", optimizer_type="sgd", params=self.model.parameters(), lr=1.0, momentum=0.9
        )
        self.scheduler = LRSchedulerRegistry.create(
            "step", lr_scheduler_type="step", optimizer=self.optimizer, last_epoch=-1, step_size=1, gamma=0.5
        )

    def set_weight(self, value: float) -> None:
        with torch.no_grad():
            self.model.weight.fill_(value)

    @property
    def weight(self) -> float:
        return self.model.weight.item()

    def advance(self) -> None:
        """One epoch of optimizer/scheduler progress."""
        self.optimizer.torch_optimizer.step()
        self.scheduler.step()

    def step(self, stopper: EarlyStopper, value: float, epoch: int) -> bool:
        return stopper.step(value, self.model, epoch, self.optimizer, self.scheduler, None)

    def restore(self, stopper: EarlyStopper):
        return stopper.restore(self.model, self.optimizer, self.scheduler, None)


@pytest.fixture
def training() -> Training:
    return Training()


def create(key: str, restore_best: bool = False, **kwargs) -> EarlyStopper:
    return EarlyStopperRegistry.create(key, early_stopper_type=key, restore_best=restore_best, **kwargs)


def run(training: Training, stopper: EarlyStopper, values: list[float]) -> list[bool]:
    stopper.start()
    return [training.step(stopper, value, epoch) for epoch, value in enumerate(values)]


###############################################################################
################################ BASIC STOPPER ################################
###############################################################################


def test_basic_stopper_stops_after_patience_epochs_without_improvement(training):
    stopper = create("basic", patience=2)

    decisions = run(training, stopper, [1.0, 0.9, 0.95, 0.96, 0.97])

    assert decisions == [False, False, False, True, True]
    assert (stopper.best_value, stopper.best_epoch) == (0.9, 1)


def test_an_improvement_resets_the_patience_counter(training):
    stopper = create("basic", patience=2)

    decisions = run(training, stopper, [1.0, 1.1, 0.8, 0.9, 0.95])

    assert decisions == [False, False, False, False, True]
    assert stopper.best_epoch == 2


def test_the_first_value_never_stops_training(training):
    assert run(training, create("basic", patience=1), [5.0]) == [False]


def test_min_delta_scales_with_the_epochs_since_the_best_value(training):
    stopper = create("basic", patience=100, min_delta=0.1)

    # best 1.0 at epoch 0: epoch 1 must beat 0.9, epoch 2 must beat 0.8
    run(training, stopper, [1.0, 0.95, 0.85])
    assert stopper.best_epoch == 0

    run(training, stopper, [1.0, 0.95, 0.79])
    assert (stopper.best_epoch, stopper.best_value) == (2, 0.79)


def test_start_discards_progress_from_a_previous_run(training):
    stopper = create("basic", patience=1)
    run(training, stopper, [1.0, 2.0, 3.0])

    stopper.start()

    assert (stopper.best_epoch, stopper.best_value, stopper.best_state) == (-1, float("inf"), None)
    assert training.step(stopper, 10.0, 0) is False


###############################################################################
################################ OTHER STOPPERS ###############################
###############################################################################


def test_the_no_op_stopper_never_stops_but_still_tracks_the_best_value(training):
    stopper = create("none")

    assert run(training, stopper, [3.0, 2.0, 5.0, 9.0, 9.0, 9.0]) == [False] * 6
    assert (stopper.best_value, stopper.best_epoch) == (2.0, 1)


def test_the_no_op_stopper_is_available_as_no_and_none():
    assert EarlyStopperRegistry.get("no") is EarlyStopperRegistry.get("none")


def test_time_stopper_stops_once_the_time_limit_is_exceeded(monkeypatch, training):
    now = [100.0]
    monkeypatch.setattr(time_stopper, "time", SimpleNamespace(time=lambda: now[0]))
    stopper = create("time", time_limit_seconds=10.0)
    stopper.start()

    now[0] = 105.0
    assert training.step(stopper, 1.0, 0) is False
    now[0] = 110.0
    assert training.step(stopper, 1.0, 1) is False  # exactly at the limit
    now[0] = 110.5
    assert training.step(stopper, 1.0, 2) is True


def test_time_stopper_measures_from_start(monkeypatch, training):
    now = [0.0]
    monkeypatch.setattr(time_stopper, "time", SimpleNamespace(time=lambda: now[0]))
    stopper = create("time", time_limit_seconds=10.0)

    now[0] = 1000.0
    stopper.start()
    now[0] = 1005.0

    assert training.step(stopper, 1.0, 0) is False


###############################################################################
############################## RESTORING BEST STATE ###########################
###############################################################################


def test_restore_returns_the_model_to_its_best_epoch(training):
    stopper = create("none", restore_best=True)
    stopper.start()

    training.set_weight(1.0)
    training.step(stopper, 5.0, 0)
    training.set_weight(2.0)
    training.step(stopper, 3.0, 1)  # best
    training.set_weight(3.0)
    training.step(stopper, 4.0, 2)

    assert training.restore(stopper) == (3.0, 1)
    assert training.weight == 2.0


def test_the_snapshot_is_independent_of_later_training(training):
    stopper = create("none", restore_best=True)
    stopper.start()
    training.set_weight(7.0)
    training.step(stopper, 1.0, 0)

    training.set_weight(-1.0)  # continued training modifies the live parameters

    training.restore(stopper)
    assert training.weight == 7.0


def test_restore_also_rewinds_optimizer_and_scheduler_state(training):
    stopper = create("none", restore_best=True)
    stopper.start()
    training.advance()
    best_lr = training.optimizer.torch_optimizer.param_groups[0]["lr"]
    training.step(stopper, 1.0, 0)

    training.advance()
    training.advance()
    assert training.optimizer.torch_optimizer.param_groups[0]["lr"] != best_lr
    training.step(stopper, 2.0, 1)

    training.restore(stopper)

    assert training.optimizer.torch_optimizer.param_groups[0]["lr"] == best_lr
    assert training.scheduler.state_dict()["last_epoch"] == 1


def test_restore_is_a_no_op_unless_requested_and_available(training):
    keep_last = create("none", restore_best=False)
    keep_last.start()
    training.set_weight(1.0)
    training.step(keep_last, 1.0, 0)
    training.set_weight(9.0)

    assert training.restore(keep_last) == (None, None)
    assert training.weight == 9.0

    never_stepped = create("none", restore_best=True)
    never_stepped.start()
    assert training.restore(never_stepped) == (None, None)
    assert training.weight == 9.0


def test_no_snapshot_is_kept_when_restoring_is_disabled(training):
    stopper = create("none", restore_best=False)
    run(training, stopper, [1.0])

    assert stopper.best_state is None
    assert stopper.best_training_state is None
