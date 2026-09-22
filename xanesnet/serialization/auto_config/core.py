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

"""Core resolution functions for automatic model and encoding configuration."""

import logging

from xanesnet.batchprocessors import BatchProcessor, BatchProcessorRegistry
from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.utils.exceptions import ConfigError

from ..config import Config
from .helpers import (
    format_value,
    requested_auto_fields,
)
from .registries import EncodingAutoResolver, ModelAutoResolver
from .statistics import SpectralStatisticsCollector, collect_spectral_statistics


def _any_auto_fields(encodings: list[dict]) -> bool:
    """Check whether any encoding (including nested concat sub-encodings) has ``"auto"`` fields.

    Args:
        encodings: List of raw encoding configuration dictionaries.

    Returns:
        ``True`` if at least one encoding or nested sub-encoding requests
        automatic resolution.
    """
    for item in encodings:
        if requested_auto_fields(item):
            return True
        if item.get("encoding_type") == "concat":
            nested = item.get("encodings", [])
            if isinstance(nested, list) and _any_auto_fields(nested):
                return True
    return False


def _resolve_auto_fields_recursive(
    encodings: list[dict],
    statistics: SpectralStatisticsCollector,
    batchprocessor: BatchProcessor,
    dataset_type: str,
) -> None:
    """Recursively resolve ``"auto"`` fields in encodings, descending into concat sub-encodings.

    Args:
        encodings: List of raw encoding configuration dictionaries (mutated in
            place).
        statistics: Pre-computed spectral statistics collector.
        batchprocessor: Batch processor used for statistics collection (for
            logging only).
        dataset_type: Dataset type string (for logging only).

    Raises:
        ConfigError: If an encoding with auto fields has no registered resolver
            or the resolver does not produce a requested field.
    """
    for item in encodings:
        if item.get("encoding_type") == "concat":
            nested = item.get("encodings", [])
            if isinstance(nested, list):
                _resolve_auto_fields_recursive(nested, statistics, batchprocessor, dataset_type)
            continue

        auto_fields = requested_auto_fields(item)
        if not auto_fields:
            continue

        encoding_type = item["encoding_type"]
        try:
            resolver = EncodingAutoResolver.get(encoding_type)
        except KeyError as exc:
            raise ConfigError(
                f"No automatic encoding config resolver registered for encoding '{encoding_type}'."
            ) from exc
        resolved_fields = resolver(item, statistics)

        for field in sorted(auto_fields):
            if field not in resolved_fields:
                raise ConfigError(f"Automatic resolver for encoding '{encoding_type}' did not produce {field}.")

            item[field] = resolved_fields[field]
            logging.info(
                "Resolved encoding '%s' field %s=%s from dataset '%s' with %s.",
                encoding_type,
                field,
                format_value(resolved_fields[field]),
                dataset_type,
                type(batchprocessor).__name__,
            )


def resolve_auto_encoding_config(config: Config, dataset: Dataset) -> Config:
    """Resolve encoding fields set to ``"auto"`` from a prepared dataset.

    The input config is expected to have passed schema-backed training
    validation.  This function performs only the dataset-dependent
    finalization for the top-level ``encodings`` list: it accumulates overall
    and per-target-site-element spectral statistics in a single batched,
    streaming pass (so the full spectral matrix is never materialized), asks the
    encoding-specific resolver for concrete values (e.g. per-point statistics
    over the training spectra), and returns a new ``Config`` without mutating
    the input config.

    For forward datasets the encoding is applied to the model target (spectra),
    so statistics are collected from target spectra.  For inverse datasets the
    encoding is applied to the spectral model input, so statistics are
    collected from input spectra.  Both prediction directions are handled
    uniformly.

    Nested encodings inside a ``concat`` encoding are resolved recursively so
    that auto fields (e.g. ``num_points`` for a Gaussian sub-encoding) are
    filled in before the concat encoding is instantiated.

    Args:
        config: Validated training configuration.
        dataset: Prepared training dataset used to derive automatic encoding
            values.

    Returns:
        New configuration with requested automatic encoding fields replaced by
        concrete values.

    Raises:
        ConfigError: If automatic fields are requested for an encoding that has
            no resolver or if the resolver does not produce a requested field.
    """
    config_raw = config.as_dict()
    encodings = config_raw.get("encodings", [])
    model_type = config_raw["model"]["model_type"]

    if not _any_auto_fields(encodings):
        return Config(config_raw)

    batchprocessor = BatchProcessorRegistry.create((dataset.dataset_type, model_type))
    statistics = collect_spectral_statistics(dataset, batchprocessor)

    _resolve_auto_fields_recursive(
        encodings,
        statistics,
        batchprocessor,
        dataset.dataset_type,
    )

    return Config(config_raw)


def resolve_auto_model_config(config: Config, dataset: Dataset, encoding: SpectraEncoding) -> Config:
    """Resolve model fields set to ``"auto"`` from a prepared dataset.

    The input config is expected to have passed schema-backed training
    validation, including validation that ``"auto"`` is used only for supported
    top-level model fields.  This function only performs the dataset-dependent
    finalization: it uses the registered batch processor for the dataset/model
    pair to prepare one sample, maps the raw input and target into the spaces
    the model consumes and predicts via the batch processor's
    :meth:`~xanesnet.batchprocessors.base.BatchProcessor.encode_input` and
    :meth:`~xanesnet.batchprocessors.base.BatchProcessor.encode_target`, asks
    the model-specific resolver for concrete dimensions using the prepared
    dataset, and returns a new ``Config`` without mutating the input config.

    The batch processor is created *with* the resolved encoding so that
    resolved input and output dimensions match the tensors the model actually
    receives at train time: for forward datasets the encoding shapes the target
    (and thus the output size), while for inverse datasets the encoding shapes
    the spectral input (and thus the input size).

    Args:
        config: Validated training configuration.
        dataset: Prepared training dataset used to derive automatic model
            values.
        encoding: Spectra encoding forwarded to the batch processor so that
            resolved dimensions match the model's encoded prediction space
            (forward) or encoded input space (inverse).

    Returns:
        New configuration with requested automatic model fields replaced by
        concrete values.

    Raises:
        ConfigError: If automatic fields are requested for a model that has no
            resolver or if the resolver does not produce a requested field.
    """
    config_raw = config.as_dict()
    model_config = config_raw["model"]
    model_type = model_config["model_type"]
    auto_fields = requested_auto_fields(model_config)

    if not auto_fields:
        return Config(config_raw)

    try:
        resolver = ModelAutoResolver.get(model_type)
    except KeyError as exc:
        raise ConfigError(f"No automatic model config resolver registered for model '{model_type}'.") from exc

    batchprocessor = BatchProcessorRegistry.create((dataset.dataset_type, model_type), encoding=encoding)
    inputs = batchprocessor.input_preparation_single(dataset, 0)
    target = batchprocessor.target_preparation_single(dataset, 0)
    element = batchprocessor.element_preparation_single(dataset, 0)
    # Map raw inputs/target into the spaces the model actually consumes and
    # predicts.  Forward processors encode the target (spectrum) and leave the
    # input untouched; inverse processors encode the input (spectrum) and leave
    # the descriptor target untouched.
    inputs = batchprocessor.encode_input(inputs, element)
    target = batchprocessor.encode_target(target, element)
    resolved_fields = resolver(inputs, target, dataset)

    for field in sorted(auto_fields):
        if field not in resolved_fields:
            raise ConfigError(f"Automatic resolver for model '{model_type}' did not produce model.{field}.")

        model_config[field] = resolved_fields[field]
        logging.info(
            "Resolved model.%s=%s from dataset '%s' with %s.",
            field,
            format_value(resolved_fields[field]),
            dataset.dataset_type,
            type(batchprocessor).__name__,
        )

    return Config(config_raw)
