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

"""Unit tests for the ``core_infer.infer`` pipeline, run against registered fakes."""

from argparse import Namespace

import pytest

from xanesnet import core_infer
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config

from . import fakes


@pytest.fixture
def config() -> Config:
    return Config(
        {
            "device": "cpu",
            "datasource": {"datasource_type": fakes.DATASOURCE_TYPE, "path": "somewhere"},
            "dataset": {"dataset_type": fakes.DATASET_TYPE, "root": "dataset_root"},
            "strategy": {"strategy_type": fakes.STRATEGY_TYPE, "option": 3},
            "inferencer": {"inferencer_type": "basic", "batch_size": 4},
            "model": {"model_type": "fake", "hidden_size": 8},
        }
    )


@pytest.fixture
def checkpoint() -> Checkpoint:
    return Checkpoint(signature=Config(fakes.SIGNATURE), model_states=[{"weight": 1}, {"weight": 2}])


def test_pipeline_stages_run_in_order(config, checkpoint, tmp_path):
    core_infer.infer(config, Namespace(), tmp_path, checkpoint)

    assert fakes.event_names() == [
        "datasource",
        "dataset",
        "prepare",
        "check_preload",
        "strategy",
        "setup_encoding",
        "setup_models",
        "set_state_dicts",
        "setup_inferencers",
        "run_inference",
    ]


def test_checkpoint_weights_are_loaded_into_the_strategy(config, checkpoint, tmp_path):
    core_infer.infer(config, Namespace(), tmp_path, checkpoint)

    assert fakes.payload("set_state_dicts") == [{"weight": 1}, {"weight": 2}]


def test_components_receive_their_configuration(config, checkpoint, tmp_path):
    core_infer.infer(config, Namespace(), tmp_path, checkpoint)

    assert fakes.payload("datasource")["path"] == "somewhere"
    assert fakes.payload("dataset")["root"] == "dataset_root"
    assert fakes.payload("setup_models").get_int("hidden_size") == 8
    assert fakes.payload("setup_inferencers") == "cpu"


def test_strategy_is_built_for_inference_only(config, checkpoint, tmp_path):
    core_infer.infer(config, Namespace(), tmp_path, checkpoint)

    kwargs = fakes.payload("strategy")
    assert kwargs["option"] == 3
    assert kwargs["inferencer_config"].get_int("batch_size") == 4
    assert kwargs["trainer_config"] is None
    assert kwargs["checkpoint_dir"] is None
    assert kwargs["tensorboard_dir"] is None


def test_predictions_are_written_below_the_run_directory(config, checkpoint, tmp_path):
    core_infer.infer(config, Namespace(), tmp_path, checkpoint)

    assert fakes.payload("run_inference") == tmp_path / "predictions"
