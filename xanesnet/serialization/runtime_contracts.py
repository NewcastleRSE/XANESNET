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

"""Runtime config contracts that are clearer in Python than JSON Schema."""

from collections.abc import Callable, Iterator
from typing import Any

from xanesnet.utils.exceptions import ConfigError

from ._types import ConfigMode, ConfigRaw

__all__ = ["validate_runtime_contracts"]

_RuntimeContract = Callable[[ConfigRaw], None]


def validate_runtime_contracts(config: ConfigRaw, mode: ConfigMode) -> None:
    """Validate mode-specific runtime contracts.

    Args:
        config: Schema-valid and defaulted configuration dictionary.
        mode: Config mode being validated.

    Raises:
        ConfigError: If a runtime-only contract is violated.
    """
    try:
        contracts = _RUNTIME_CONTRACTS_BY_MODE[mode]
    except KeyError as exc:
        raise ConfigError(f"Unknown config validation mode: {mode!r}.") from exc

    for contract in contracts:
        contract(config)


def _require_registered_batch_processor(config: ConfigRaw) -> None:
    """Validate that the dataset/model pair has a registered batch processor.

    Args:
        config: Schema-valid training or merged inference configuration.

    Raises:
        ConfigError: If no batch processor is registered for the configured
            dataset and model types.
    """
    dataset_type = _section_value(config, "dataset", "dataset_type")
    model_type = _section_value(config, "model", "model_type")
    if dataset_type is None or model_type is None:
        return

    from xanesnet.batchprocessors import BatchProcessorRegistry

    try:
        BatchProcessorRegistry.get((str(dataset_type), str(model_type)))
    except KeyError as exc:
        raise ConfigError(
            f"No batch processor registered for dataset '{dataset_type}' and model '{model_type}'."
        ) from exc


def _require_concrete_inference_model(config: ConfigRaw) -> None:
    """Validate that merged inference model values are concrete.

    Args:
        config: Schema-valid merged inference configuration.

    Raises:
        ConfigError: If the model section still contains an ``"auto"`` token.
    """
    model_config = config.get("model")
    for path in _auto_token_paths(model_config, ("model",)):
        raise ConfigError(
            f"Inference config contains unresolved automatic model value at {path}. "
            "Checkpoint signatures must provide concrete model dimensions."
        )


def _require_concrete_inference_encodings(config: ConfigRaw) -> None:
    """Validate that merged inference encoding values are concrete.

    Args:
        config: Schema-valid merged inference configuration.

    Raises:
        ConfigError: If the encodings section still contains an ``"auto"`` token.
    """
    encodings_config = config.get("encodings")
    for path in _auto_token_paths(encodings_config, ("encodings",)):
        raise ConfigError(
            f"Inference config contains unresolved automatic encoding value at {path}. "
            "Checkpoint signatures must provide concrete encoding dimensions."
        )


def _require_ensemble_inferencer_for_ensemble_strategies(config: ConfigRaw) -> None:
    """Validate ensemble-strategy inference runner selection.

    Args:
        config: Schema-valid merged inference configuration.

    Raises:
        ConfigError: If an ensemble strategy is paired with a non-ensemble
            inferencer.
    """
    strategy_type = _section_value(config, "strategy", "strategy_type")
    inferencer_type = _section_value(config, "inferencer", "inferencer_type")
    if strategy_type in {"deep_ensemble", "bootstrap"} and inferencer_type != "ensemble":
        raise ConfigError(f"Inference strategy '{strategy_type}' requires inferencer 'ensemble'.")


def _section_value(config: ConfigRaw, section: str, key: str) -> Any:
    """Return a nested config value from a section when present.

    Args:
        config: Configuration dictionary to inspect.
        section: Top-level section name.
        key: Key within ``section`` to read.

    Returns:
        Nested value or ``None`` when the section is not a dictionary.
    """
    section_value = config.get(section)
    if not isinstance(section_value, dict):
        return None
    return section_value.get(key)


def _auto_token_paths(value: Any, path: tuple[str, ...]) -> Iterator[str]:
    """Yield dotted paths whose value is the case-insensitive ``auto`` token.

    Args:
        value: Config value to inspect.
        path: Path components leading to ``value``.

    Yields:
        Dotted config paths containing ``auto``.
    """
    if isinstance(value, str) and value.lower() == "auto":
        yield ".".join(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _auto_token_paths(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _auto_token_paths(child, (*path, str(index)))


_RUNTIME_CONTRACTS_BY_MODE: dict[ConfigMode, tuple[_RuntimeContract, ...]] = {
    "train": (_require_registered_batch_processor,),
    "infer": (
        _require_registered_batch_processor,
        _require_concrete_inference_model,
        _require_concrete_inference_encodings,
        _require_ensemble_inferencer_for_ensemble_strategies,
    ),
    "analyze": (),
}
