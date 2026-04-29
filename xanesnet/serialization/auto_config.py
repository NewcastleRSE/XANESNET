# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Automatic model-configuration resolution from prepared datasets."""

import logging
from collections.abc import Callable
from typing import Any

import torch

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.datasets import Dataset
from xanesnet.utils.exceptions import ConfigError

from .config import Config

AUTO_VALUE = "auto"
AutoResolver = Callable[[dict[str, Any], Any], dict[str, Any]]


def resolve_auto_model_config(config: Config, dataset: Dataset) -> Config:
    """Resolve model fields set to ``"auto"`` from a prepared dataset.

    The input config is expected to have passed training validation, including
    validation that ``"auto"`` is used only for supported top-level model
    fields. This function only performs the dataset-dependent finalization: it
    uses the registered batch processor for the dataset/model pair to prepare
    one sample, asks the model-specific resolver for concrete dimensions, and
    returns a new ``Config`` without mutating the input config.

    Args:
        config: Validated training configuration.
        dataset: Prepared training dataset used to derive automatic model
            values.

    Returns:
        New configuration with requested automatic model fields replaced by
        concrete values.

    Raises:
        ConfigError: If automatic fields are requested for a model that has no
            resolver or if the resolver cann derive a requested value from the prepared sample.
    """
    config_raw = config.as_dict()
    model_config = config_raw["model"]
    model_type = model_config["model_type"]
    auto_fields = _requested_auto_fields(model_config)

    if not auto_fields:
        return Config(config_raw)

    resolver = _resolver_for(model_type)

    batchprocessor_type = BatchProcessorRegistry.get(dataset.dataset_type, model_type)
    batchprocessor = batchprocessor_type()
    inputs = batchprocessor.input_preparation_single(dataset, 0)
    target = batchprocessor.target_preparation_single(dataset, 0)
    resolved_fields = resolver(inputs, target)

    for field in sorted(auto_fields):
        if field not in resolved_fields:
            raise ConfigError(f"Automatic resolver for model '{model_type}' did not produce model.{field}.")

        model_config[field] = resolved_fields[field]
        logging.info(
            "Resolved model.%s=%s from dataset '%s' with %s.",
            field,
            resolved_fields[field],
            dataset.dataset_type,
            type(batchprocessor).__name__,
        )

    return Config(config_raw)


def _requested_auto_fields(model_config: dict[str, Any]) -> set[str]:
    """Return top-level model fields whose value is ``"auto"``."""
    return {key for key, value in model_config.items() if _is_auto(value)}


def _resolver_for(model_type: str) -> AutoResolver:
    """Return the automatic-field resolver for ``model_type``.

    Args:
        model_type: Model registry key.

    Returns:
        Resolver function for the model type.

    Raises:
        ConfigError: If no automatic-field resolver is registered for the
            model type.
    """
    try:
        return MODEL_AUTO_RESOLVERS[model_type]
    except KeyError as exc:
        raise ConfigError(f"No automatic model config resolver registered for model '{model_type}'.") from exc


def _resolve_descriptor_io(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve descriptor-vector input size and target spectrum length."""
    return {
        "in_size": _last_dim("input 'x'", _required_tensor(inputs, "x")),
        "out_size": _last_dim("target", target),
    }


def _resolve_mlp(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve MLP input and output dimensions."""
    return _resolve_descriptor_io(inputs, target)


def _resolve_lstm(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve LSTM input and output dimensions."""
    return _resolve_descriptor_io(inputs, target)


def _resolve_envembed(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve EnvEmbed descriptor and spectral-basis dimensions."""
    return {
        "in_size": _last_dim("input 'descriptor_features'", _required_tensor(inputs, "descriptor_features")),
        "kgroups": _kgroups_from_basis(inputs.get("basis")),
    }


def _resolve_schnet(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve SchNet output dimension."""
    return {"reduce_channels_2": _last_dim("target", target)}


def _resolve_dimenet(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve DimeNet output dimension."""
    return {"out_channels": _last_dim("target", target)}


def _resolve_dimenet_pp(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve DimeNet++ output dimension."""
    return {"out_channels": _last_dim("target", target)}


def _resolve_gemnet(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve GemNet output dimension."""
    return {"num_targets": _last_dim("target", target)}


def _resolve_gemnet_oc(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve GemNet-OC output dimension."""
    return {"num_targets": _last_dim("target", target)}


def _resolve_e3ee(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve E3EE output dimension."""
    return {"out_size": _last_dim("target", target)}


def _resolve_e3ee_full(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve E3EEFull output dimension."""
    return {"out_size": _last_dim("target", target)}


MODEL_AUTO_RESOLVERS: dict[str, AutoResolver] = {
    "mlp": _resolve_mlp,
    "lstm": _resolve_lstm,
    "envembed": _resolve_envembed,
    "schnet": _resolve_schnet,
    "dimenet": _resolve_dimenet,
    "dimenet++": _resolve_dimenet_pp,
    "gemnet": _resolve_gemnet,
    "gemnet_oc": _resolve_gemnet_oc,
    "e3ee": _resolve_e3ee,
    "e3ee_full": _resolve_e3ee_full,
}


def _is_auto(value: Any) -> bool:
    """Return whether ``value`` requests automatic resolution."""
    return isinstance(value, str) and value.lower() == AUTO_VALUE


def _required_tensor(inputs: dict[str, Any], key: str) -> torch.Tensor:
    """Return a tensor input by key.

    Args:
        inputs: Model input dictionary returned by a batch processor.
        key: Required input key.

    Returns:
        Tensor stored at ``key``.

    Raises:
        ConfigError: If ``key`` is missing or does not contain a tensor.
    """
    value = inputs.get(key)
    if not isinstance(value, torch.Tensor):
        raise ConfigError(f"Cannot derive automatic model configuration because input '{key}' is not a tensor.")
    return value


def _last_dim(name: str, value: Any) -> int:
    """Return the final dimension of a tensor.

    Args:
        name: Human-readable tensor name used in error messages.
        value: Tensor whose final dimension should be used.

    Returns:
        Size of the final tensor dimension.

    Raises:
        ConfigError: If ``value`` is not a tensor or is a scalar tensor.
    """
    if not isinstance(value, torch.Tensor):
        raise ConfigError(f"Cannot derive automatic model configuration because {name} is not a tensor.")
    if value.ndim == 0:
        raise ConfigError(f"Cannot derive automatic model configuration from scalar {name} tensor.")
    return int(value.shape[-1])


def _kgroups_from_basis(basis: Any) -> list[int]:
    """Derive EnvEmbed coefficient group sizes from a spectral basis.

    Args:
        basis: Spectral basis object returned by the EnvEmbed batch processor.

    Returns:
        List of coefficient counts, one per spectral-basis width group.

    Raises:
        ConfigError: If the basis does not expose compatible ``Phi`` and
            ``widths_eV`` attributes.
    """
    phi = getattr(basis, "Phi", None)
    widths_eV = getattr(basis, "widths_eV", None)
    if not isinstance(phi, torch.Tensor):
        raise ConfigError("Cannot derive model.kgroups because the batch processor did not provide a spectral basis.")
    if not isinstance(widths_eV, list) or len(widths_eV) == 0:
        raise ConfigError("Cannot derive model.kgroups because the spectral basis has no width groups.")

    num_groups = len(widths_eV)
    num_coefficients = int(phi.shape[1])
    if num_coefficients % num_groups != 0:
        raise ConfigError(
            "Cannot derive model.kgroups because the spectral basis coefficient count is not divisible by "
            "the number of width groups."
        )

    return [num_coefficients // num_groups] * num_groups
