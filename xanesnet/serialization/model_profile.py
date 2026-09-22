# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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
#   ...

"""Model profiling helpers for training runs."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torchinfo import summary

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model


def build_model_profile(
    model: Model,
    dataset: Dataset,
    device: str | torch.device,
    peak_gpu_memory_allocated_mb: float | None,
    encoding: SpectraEncoding | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable model profile metadata.

    Args:
        model: Model to profile.
        dataset: Dataset used to prepare a representative ``torchinfo`` input.
        device: Configured training device.
        peak_gpu_memory_allocated_mb: Peak GPU memory allocated during the
            dry-run training epoch, or ``None`` when CUDA memory was not tracked.
        encoding: Prepared spectra encoding used by the model, or ``None`` when
            the model consumes raw spectra.

    Returns:
        Dictionary ready to save as ``model_profile.json``.
    """
    summary_result = create_model_summary(model, dataset, verbose=0, encoding=encoding)
    total_param_bytes = int(summary_result.total_param_bytes)

    return {
        "model_architecture": str(summary_result),
        "total_params": int(summary_result.total_params),
        "trainable_params": int(summary_result.trainable_params),
        "total_param_bytes": total_param_bytes,
        "model_size_mb": _bytes_to_mb(total_param_bytes),
        "device": str(device),
        "peak_gpu_memory_allocated_mb": peak_gpu_memory_allocated_mb,
    }


def _prepare_torchinfo_inputs(
    model: Model, inputs: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], Callable[..., Any]]:
    """Filter non-tensor inputs and wrap ``model.forward`` if needed.

    Args:
        model: Model to summarize.
        inputs: Raw input dict from the batch processor.

    Returns:
        ``(tensor_inputs, original_forward)`` — the filtered kwargs for
        ``torchinfo`` and the original ``model.forward`` that must be
        restored after ``torchinfo.summary`` completes.
    """
    tensor_inputs: dict[str, torch.Tensor] = {}
    extra_kwargs: dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            tensor_inputs[k] = v
        else:
            extra_kwargs[k] = v

    _forward = model.forward
    if extra_kwargs:
        model.forward = lambda *a, **kw: _forward(*a, **kw, **extra_kwargs)  # type: ignore[method-assign]

    return tensor_inputs, _forward


def create_model_summary(
    model: Model,
    dataset: Dataset,
    verbose: int | None = None,
    encoding: SpectraEncoding | None = None,
) -> Any:
    """Create a ``torchinfo`` summary for a model and dataset sample.

    Args:
        model: Model to summarize.
        dataset: Dataset used to prepare one representative input sample.
        verbose: Optional ``torchinfo.summary`` verbosity. When ``None``, the
            torchinfo default is used.
        encoding: Prepared spectra encoding used by the model, or ``None`` when
            the model consumes raw spectra.

    Returns:
        The ``torchinfo`` summary result.
    """
    batchprocessor = BatchProcessorRegistry.create(
        (dataset.dataset_type, model.model_type),
        encoding=encoding,
    )
    inputs = batchprocessor.input_preparation_single(dataset, 0)
    elements = batchprocessor.element_preparation_single(dataset, 0)
    inputs = batchprocessor.encode_input(inputs, elements)

    tensor_inputs, _forward = _prepare_torchinfo_inputs(model, inputs)
    try:
        if verbose is None:
            return summary(model, input_data=tensor_inputs)
        return summary(model, input_data=tensor_inputs, verbose=verbose)
    finally:
        model.forward = _forward


def reset_peak_memory_stats(device: str | torch.device) -> bool:
    """Reset peak CUDA memory stats for ``device`` when it is a CUDA device.

    Args:
        device: Configured training device.

    Returns:
        ``True`` if CUDA memory tracking was reset, otherwise ``False``.
    """
    cuda_device = _cuda_device(device)
    if cuda_device is None:
        return False

    torch.cuda.reset_peak_memory_stats(cuda_device)
    return True


def get_peak_memory_allocated_mb(device: str | torch.device) -> float | None:
    """Return peak CUDA memory allocated for ``device`` in MB.

    Args:
        device: Configured training device.

    Returns:
        Peak CUDA memory allocated in MB, or ``None`` when ``device`` is not a
        CUDA device or CUDA is unavailable.
    """
    cuda_device = _cuda_device(device)
    if cuda_device is None:
        return None

    torch.cuda.synchronize(cuda_device)
    return _bytes_to_mb(torch.cuda.max_memory_allocated(cuda_device))


def save_model_profile(save_dir: Path, model_profile: dict[str, Any]) -> tuple[Path, Path]:
    """Save model profile metadata as JSON and human-readable text.

    Args:
        save_dir: Root run directory.
        model_profile: JSON-serializable profile metadata.

    Returns:
        Paths to the written ``model_profile.json`` and
        ``model_profile_readable.txt`` files.
    """
    profile_save_path = save_dir / "model_profile.json"
    readable_save_path = save_dir / "model_profile_readable.txt"

    with profile_save_path.open("w", encoding="utf-8") as profile_file:
        json.dump(model_profile, profile_file, indent=2)
        profile_file.write("\n")

    with readable_save_path.open("w", encoding="utf-8") as readable_file:
        readable_file.write("Model Profile\n")
        readable_file.write("=============\n\n")

        readable_file.write("Model Architecture\n")
        readable_file.write("------------------\n")
        readable_file.write(f"{model_profile['model_architecture']}\n\n")

        readable_file.write("Metadata\n")
        readable_file.write("--------\n")
        for key, value in model_profile.items():
            if key == "model_architecture":
                continue
            readable_file.write(f"{key}: {value}\n")

    return profile_save_path, readable_save_path


def _cuda_device(device: str | torch.device) -> torch.device | None:
    """Return ``device`` as CUDA device when CUDA memory can be tracked."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None

    return torch_device


def _bytes_to_mb(value: int) -> float:
    """Convert bytes to MB."""
    return value / (1024 * 1024)
