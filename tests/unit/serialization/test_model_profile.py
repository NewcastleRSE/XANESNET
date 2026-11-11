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

"""Unit tests for model-profile persistence and device-aware memory helpers."""

import json

import pytest
import torch

from xanesnet.serialization.model_profile import (
    get_peak_memory_allocated_mb,
    reset_peak_memory_stats,
    save_model_profile,
)


@pytest.fixture
def profile() -> dict:
    return {
        "model_architecture": "Linear(3, 2)\n  weights: 8",
        "total_params": 8,
        "trainable_params": 8,
        "total_param_bytes": 32,
        "model_size_mb": 3.0e-05,
        "device": "cpu",
        "peak_gpu_memory_allocated_mb": None,
    }


###############################################################################
############################### save_model_profile ############################
###############################################################################


def test_both_report_files_are_written(tmp_path, profile):
    json_path, text_path = save_model_profile(tmp_path, profile)
    assert (json_path, text_path) == (tmp_path / "model_profile.json", tmp_path / "model_profile_readable.txt")
    assert json_path.is_file() and text_path.is_file()


def test_json_report_round_trips_the_profile(tmp_path, profile):
    json_path, _ = save_model_profile(tmp_path, profile)
    assert json.loads(json_path.read_text()) == profile


def test_readable_report_lists_architecture_first_then_metadata(tmp_path, profile):
    _, text_path = save_model_profile(tmp_path, profile)
    text = text_path.read_text()

    assert text.startswith("Model Profile\n=============")
    assert text.index("Model Architecture") < text.index("Linear(3, 2)") < text.index("Metadata")
    assert "total_params: 8" in text
    assert "peak_gpu_memory_allocated_mb: None" in text
    assert text.count("Linear(3, 2)") == 1, "the architecture must not be repeated in the metadata block"


###############################################################################
############################### MEMORY TRACKING ###############################
###############################################################################


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_memory_tracking_is_disabled_for_cpu_devices(device):
    assert reset_peak_memory_stats(device) is False
    assert get_peak_memory_allocated_mb(device) is None


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without CUDA")
def test_memory_tracking_is_disabled_for_cuda_devices_when_cuda_is_unavailable():
    assert reset_peak_memory_stats("cuda") is False
    assert get_peak_memory_allocated_mb("cuda:0") is None


def test_memory_tracking_converts_bytes_to_megabytes_on_cuda(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: calls.append(f"reset:{device}"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(f"sync:{device}"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 3 * 1024 * 1024)

    assert reset_peak_memory_stats("cuda:1") is True
    assert get_peak_memory_allocated_mb("cuda:1") == pytest.approx(3.0)
    assert calls == ["reset:cuda:1", "sync:cuda:1"]
