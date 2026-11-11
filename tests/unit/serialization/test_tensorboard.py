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

"""Unit tests for the TensorBoard logging singleton."""

import logging
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import TensorBoardLogger, tb_logger


@pytest.fixture(autouse=True)
def _clean_logger():
    """The logger is a process-wide singleton: close it and reset opt-in flags around every test."""
    tb_logger.close()
    tb_logger.set_log_weights(False)
    tb_logger.set_config(Config({}))
    yield
    tb_logger.close()
    tb_logger.set_log_weights(False)
    tb_logger.set_config(Config({}))


def _read(log_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(log_dir), size_guidance={"tensors": 0, "histograms": 0, "scalars": 0})
    accumulator.Reload()
    return accumulator


def _scalars(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    return [(event.step, event.value) for event in _read(log_dir).Scalars(tag)]


class _Scale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return self.linear(x) * scale


###############################################################################
################################## LIFECYCLE ##################################
###############################################################################


def test_logger_is_a_singleton():
    assert TensorBoardLogger() is tb_logger


def test_logger_is_disabled_until_a_run_is_started():
    assert tb_logger.enabled is False


def test_logging_before_a_run_is_a_silent_noop():
    tb_logger.log_scalar("a", 1.0, 0)
    tb_logger.log_learning_rate(0, 0.1)
    tb_logger.log_epoch_metrics(0, 1.0, 0.0, 1.0)
    tb_logger.log_text("t", "x", 0)
    tb_logger.log_final_metrics({"m": 1.0})
    tb_logger.flush()


def test_writer_access_requires_a_started_run():
    with pytest.raises(RuntimeError, match="not initialized"):
        tb_logger.writer


def test_new_run_enables_logging_and_creates_event_files(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.log_scalar("x", 1.0, 0)
    tb_logger.flush()

    assert tb_logger.enabled is True
    assert list(tmp_path.glob("events.out.tfevents.*"))


def test_close_disables_logging_again(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.close()

    assert tb_logger.enabled is False
    tb_logger.log_scalar("x", 1.0, 0)  # must not raise


def test_closing_twice_is_harmless(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.close()
    tb_logger.close()


def test_starting_a_new_run_redirects_output_to_the_new_directory(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    tb_logger.new_run(first)
    tb_logger.log_scalar("x", 1.0, 0)
    tb_logger.new_run(second)
    tb_logger.log_scalar("y", 2.0, 0)
    tb_logger.flush()

    assert _scalars(first, "x") == [(0, 1.0)]
    assert _scalars(second, "y") == [(0, 2.0)]
    assert "y" not in _read(first).Tags()["scalars"]


###############################################################################
################################### METRICS ###################################
###############################################################################


def test_log_scalar_records_value_at_step(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.log_scalar("custom/metric", 0.5, 3)
    tb_logger.flush()
    assert _scalars(tmp_path, "custom/metric") == [(3, 0.5)]


def test_epoch_metrics_without_validation(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.log_epoch_metrics(2, train_loss=1.0, train_regularization=0.25, train_total=1.25)
    tb_logger.flush()

    tags = set(_read(tmp_path).Tags()["scalars"])
    assert tags == {"loss/train", "regularization/train", "total/train"}
    assert _scalars(tmp_path, "total/train") == [(2, 1.25)]


def test_epoch_metrics_with_validation(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.log_epoch_metrics(
        0, 1.0, 0.0, 1.0, valid_loss=2.0, valid_regularization=0.5, valid_total=2.5
    )
    tb_logger.flush()

    assert _scalars(tmp_path, "loss/valid") == [(0, 2.0)]
    assert _scalars(tmp_path, "regularization/valid") == [(0, 0.5)]
    assert _scalars(tmp_path, "total/valid") == [(0, 2.5)]


def test_learning_rate_is_logged_under_the_other_group(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.log_learning_rate(4, 0.01)
    tb_logger.flush()
    assert _scalars(tmp_path, "other/learning_rate") == [(4, pytest.approx(0.01))]


def test_text_and_histogram_logging(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.log_text("note", "hello", 0)
    tb_logger.log_histogram("dist", torch.randn(100), 0)
    tb_logger.flush()

    tags = _read(tmp_path).Tags()
    assert "note/text_summary" in tags["tensors"]
    assert "dist" in tags["histograms"]


###############################################################################
############################ WEIGHT / GRADIENT HISTOGRAMS #####################
###############################################################################


def _model_with_gradients() -> torch.nn.Module:
    model = torch.nn.Linear(3, 2)
    model(torch.randn(4, 3)).sum().backward()
    return model


def test_weight_histograms_are_opt_in(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.log_model_weights(0, _model_with_gradients())
    tb_logger.flush()
    assert _read(tmp_path).Tags()["histograms"] == []


def test_weight_and_gradient_histograms_when_enabled(tmp_path):
    tb_logger.set_log_weights(True)
    tb_logger.new_run(tmp_path)
    tb_logger.log_model_weights(0, _model_with_gradients())
    tb_logger.flush()

    assert set(_read(tmp_path).Tags()["histograms"]) == {
        "weights/weight",
        "weights/bias",
        "gradients/weight",
        "gradients/bias",
    }


def test_gradient_histograms_are_skipped_for_parameters_without_gradients(tmp_path):
    tb_logger.set_log_weights(True)
    tb_logger.new_run(tmp_path)
    tb_logger.log_model_weights(0, torch.nn.Linear(3, 2))
    tb_logger.flush()

    assert set(_read(tmp_path).Tags()["histograms"]) == {"weights/weight", "weights/bias"}


###############################################################################
################################# HYPERPARAMETERS #############################
###############################################################################


def test_config_is_written_as_a_hyperparameter_table(tmp_path):
    tb_logger.set_config(Config({"trainer": {"epochs": 3, "lr_scheduler": {"type": "none"}}, "layers": [1, 2]}))
    tb_logger.new_run(tmp_path)
    tb_logger.flush()

    accumulator = _read(tmp_path)
    assert "hyperparameters/text_summary" in accumulator.Tags()["tensors"]
    text = accumulator.Tensors("hyperparameters/text_summary")[0].tensor_proto.string_val[0].decode()
    assert "`trainer.epochs` | `3`" in text
    assert "`trainer.lr_scheduler.type` | `none`" in text
    assert "`layers` | `[1, 2]`" in text


def test_final_metrics_are_logged_next_to_hyperparameters(tmp_path):
    tb_logger.set_config(Config({"seed": 1, "model": {"kind": "mlp"}, "skipped": None}))
    tb_logger.new_run(tmp_path)
    tb_logger.log_final_metrics({"hparam/final_loss": 0.5})
    tb_logger.flush()

    assert list(tmp_path.glob("events.out.tfevents.*")), "events must be written to the run directory itself"
    assert not [p for p in tmp_path.iterdir() if p.is_dir()], "run_name='.' must not create a sub-directory"


###############################################################################
################################### MODEL GRAPH ###############################
###############################################################################


def test_model_graph_is_logged_with_bound_non_tensor_inputs(tmp_path):
    tb_logger.new_run(tmp_path)
    tb_logger.log_model_graph(_Scale(), {"x": torch.randn(2, 3), "scale": 2.0})
    tb_logger.flush()
    assert _read(tmp_path).Graph().node


def test_model_graph_failures_degrade_to_a_warning(tmp_path, caplog):
    tb_logger.new_run(tmp_path)
    with caplog.at_level(logging.WARNING):
        tb_logger.log_model_graph(_Scale(), {"scale": 2.0})  # the required tensor input is missing
    assert "Skipping TensorBoard graph logging" in caplog.text


def test_model_graph_without_tensor_inputs_is_skipped(tmp_path, caplog):
    class NoTensors(torch.nn.Module):
        def forward(self, scale: float = 1.0) -> float:
            return scale

    tb_logger.new_run(tmp_path)
    with caplog.at_level(logging.WARNING):
        tb_logger.log_model_graph(NoTensors(), {"scale": 2.0})
    assert "no tensor inputs" in caplog.text


def test_model_graph_logging_is_a_noop_when_disabled():
    tb_logger.log_model_graph(_Scale(), {"x": torch.randn(2, 3)})
