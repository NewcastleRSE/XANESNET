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

"""Unit tests for the XANESNET learning-rate scheduler wrappers."""

import pytest
import torch

from xanesnet.components import LRSchedulerRegistry, OptimizerRegistry, Scheduler
from xanesnet.serialization.config import Config

BASE_LR = 1.0

# scheduler key -> (constructor arguments, learning rate seen by each epoch)
SCHEDULES = {
    "step": ({"step_size": 2, "gamma": 0.5}, [1.0, 1.0, 0.5, 0.5, 0.25, 0.25]),
    "multistep": ({"milestones": [1, 3], "gamma": 0.5}, [1.0, 0.5, 0.5, 0.25, 0.25]),
    "exponential": ({"gamma": 0.5}, [1.0, 0.5, 0.25, 0.125]),
    "linear": ({"start_factor": 0.1, "end_factor": 1.0, "total_iters": 4}, [0.1, 0.325, 0.55, 0.775, 1.0, 1.0]),
    "constant": ({"factor": 0.5, "total_iters": 2}, [0.5, 0.5, 1.0, 1.0]),
    "none": ({}, [1.0] * 4),
    "no": ({}, [1.0] * 4),
}


@pytest.fixture
def optimizer():
    parameter = torch.nn.Parameter(torch.zeros(1))
    return OptimizerRegistry.create("sgd", optimizer_type="sgd", params=[parameter], lr=BASE_LR)


def make(kind: str, optimizer) -> Scheduler:
    return LRSchedulerRegistry.create(
        kind, lr_scheduler_type=kind, optimizer=optimizer, last_epoch=-1, **SCHEDULES[kind][0]
    )


def trajectory(scheduler: Scheduler, optimizer, epochs: int) -> list[float]:
    """Learning rate seen by each epoch, stepping the scheduler once per epoch."""
    rates = []
    for _ in range(epochs):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return rates


@pytest.mark.parametrize("kind", SCHEDULES)
def test_registry_creates_a_scheduler_that_records_its_signature(kind, optimizer):
    scheduler = make(kind, optimizer)

    assert isinstance(scheduler, Scheduler)
    assert scheduler.torch_scheduler is scheduler
    assert isinstance(scheduler.signature, Config)
    assert scheduler.signature.as_dict() == {"lr_scheduler_type": kind}


@pytest.mark.parametrize("kind", SCHEDULES)
def test_state_dict_excludes_xanesnet_metadata(kind, optimizer):
    state = make(kind, optimizer).state_dict()
    assert "_signature" not in state
    assert "lr_scheduler_type" not in state


@pytest.mark.parametrize("kind", SCHEDULES)
def test_constructor_arguments_reach_the_torch_scheduler(kind, optimizer):
    """Each wrapper must forward its arguments correctly; the expected rates follow from them."""
    expected = SCHEDULES[kind][1]
    assert trajectory(make(kind, optimizer), optimizer, len(expected)) == pytest.approx(expected)


def test_scheduler_state_survives_a_restore_on_a_fresh_optimizer(optimizer):
    original = make("step", optimizer)
    for _ in range(3):
        optimizer.step()
        original.step()

    resumed_optimizer = OptimizerRegistry.create(
        "sgd", optimizer_type="sgd", params=[torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR
    )
    resumed = make("step", resumed_optimizer)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed.load_state_dict(original.state_dict())

    assert trajectory(resumed, resumed_optimizer, 4) == pytest.approx(trajectory(original, optimizer, 4))
