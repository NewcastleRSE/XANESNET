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

"""Unit tests for run metadata collection (software, hardware and git information)."""

import subprocess

import pytest
import torch

import xanesnet
from xanesnet.serialization import metadata
from xanesnet.serialization.metadata import get_hardware_info, get_software_info, write_run_metadata


def _parse_info_file(path) -> dict[str, str]:
    """Parse the ``key = value`` info format."""
    parsed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(" = ")
        parsed[key] = value
    return parsed


###############################################################################
################################ SOFTWARE INFO ################################
###############################################################################


def test_software_info_reports_package_and_runtime_versions():
    info = get_software_info("train", ["-i", "config.yaml"])

    assert info["package"] == "XANESNET"
    assert info["version"] == xanesnet.__version__
    assert info["workflow"] == "train"
    assert info["pytorch_version"] == torch.__version__
    assert info["python_version"].count(".") >= 1


def test_command_line_is_shell_quoted():
    info = get_software_info("infer", ["-i", "my config.yaml", "-n", "a'b"])
    assert info["command_line_args"] == "-i 'my config.yaml' -n 'a'\"'\"'b'"


@pytest.mark.parametrize("args", [None, []])
def test_missing_command_line_is_reported_as_unknown(args):
    assert get_software_info("train", args)["command_line_args"] == "unknown"


def test_software_info_always_contains_git_fields():
    info = get_software_info("train")
    assert {"git_commit", "git_branch", "git_dirty"} <= set(info)


def test_git_fields_degrade_to_unknown_when_git_is_unavailable(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(metadata.subprocess, "run", no_git)

    info = get_software_info("train")

    assert (info["git_commit"], info["git_branch"], info["git_dirty"]) == ("unknown",) * 3


@pytest.mark.parametrize("error", [subprocess.CalledProcessError(128, "git"), subprocess.TimeoutExpired("git", 5)])
def test_git_failures_degrade_to_unknown(monkeypatch, error):
    def failing_git(*args, **kwargs):
        raise error

    monkeypatch.setattr(metadata.subprocess, "run", failing_git)
    assert get_software_info("train")["git_commit"] == "unknown"


def _fake_git(outputs: dict[str, str]):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=outputs[command[1]] + "\n", stderr="")

    return run


@pytest.mark.parametrize("status, dirty", [("", "false"), (" M file.py", "true")])
def test_git_dirty_flag_follows_porcelain_status(monkeypatch, status, dirty):
    monkeypatch.setattr(
        metadata.subprocess, "run", _fake_git({"rev-parse": "abc123", "branch": "main", "status": status})
    )
    info = get_software_info("train")
    assert (info["git_commit"], info["git_branch"], info["git_dirty"]) == ("abc123", "main", dirty)


def test_detached_head_is_reported_as_detached(monkeypatch):
    monkeypatch.setattr(metadata.subprocess, "run", _fake_git({"rev-parse": "abc", "branch": "", "status": ""}))
    assert get_software_info("train")["git_branch"] == "detached"


###############################################################################
################################ HARDWARE INFO ################################
###############################################################################


def test_hardware_info_contains_the_expected_fields():
    info = get_hardware_info()
    expected = {"os", "kernel", "platform", "architecture", "cpu_model", "cpu_count", "ram_bytes", "cuda_available"}
    assert expected <= set(info)
    assert info["cuda_available"] in {"true", "false", "unknown"}
    assert all(isinstance(value, str) and value for value in info.values())


def test_hardware_info_reports_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert get_hardware_info()["cuda_visible_devices"] == "0,1"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert get_hardware_info()["cuda_visible_devices"] == "unknown"


def test_cuda_probing_errors_do_not_break_hardware_collection(monkeypatch):
    def broken() -> bool:
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(torch.cuda, "is_available", broken)

    info = get_hardware_info()

    assert info["cuda_available"] == "unknown"
    assert info["gpu_count"] == "unknown"


def test_gpu_details_are_listed_per_device(monkeypatch):
    class Props:
        name, total_memory, major, minor = "FakeGPU", 1024, 8, 6

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: Props)

    info = get_hardware_info()

    assert info["cuda_available"] == "true"
    assert info["gpu_count"] == "2"
    assert info["gpu_1_name"] == "FakeGPU"
    assert info["gpu_1_memory_bytes"] == "1024"
    assert info["gpu_1_compute_capability"] == "8.6"


###############################################################################
############################### write_run_metadata ############################
###############################################################################


def test_write_run_metadata_creates_both_info_files(tmp_path):
    software_path, hardware_path = write_run_metadata(tmp_path, mode="train", command_line_args=["-i", "c.yaml"])

    assert software_path == tmp_path / "software.info"
    assert hardware_path == tmp_path / "hardware.info"
    software = _parse_info_file(software_path)
    assert software["workflow"] == "train"
    assert software["command_line_args"] == "-i c.yaml"
    assert "os" in _parse_info_file(hardware_path)


def test_write_run_metadata_accepts_string_paths(tmp_path):
    software_path, _ = write_run_metadata(str(tmp_path), mode="analyze")
    assert software_path.parent == tmp_path


def test_write_run_metadata_requires_an_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
        write_run_metadata(tmp_path / "missing", mode="train")
    assert not (tmp_path / "missing").exists()


def test_write_run_metadata_rejects_files(tmp_path):
    file = tmp_path / "file"
    file.write_text("x")
    with pytest.raises(FileNotFoundError):
        write_run_metadata(file, mode="train")
