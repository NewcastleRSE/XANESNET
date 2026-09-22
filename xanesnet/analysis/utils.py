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

"""Shared helpers for the analysis pipeline."""

from collections.abc import Iterable, Iterator, Mapping
from numbers import Integral
from typing import Any, TypeGuard

import numpy as np
import torch

ScalarValue = int | float | np.integer | np.floating
VectorValue = list | tuple | np.ndarray | torch.Tensor
SampleKey = tuple[str, int | None]

SAMPLE_METADATA_KEYS: set[str] = {"sample_id", "target_site_index"}

SPECTRUM_KEYS: set[str] = {"prediction", "target"}


def one_line_label(lines: Iterable[str]) -> str:
    """Join non-empty display-label parts with one shared separator.

    Args:
        lines: Label parts in display order.

    Returns:
        A single label with non-empty parts separated by ``" | "``.
    """
    return " | ".join(line.strip() for line in lines if line.strip())


def sample_key(record: Mapping[str, Any]) -> SampleKey:
    """Return the identity of one prediction or collector record.

    The target-site index is part of the identity because one structure can
    produce several prediction records.

    Args:
        record: Prediction or collector record containing ``sample_id`` and
            optionally ``target_site_index``.

    Returns:
        ``(sample_id, target_site_index)`` with a normalized string ID and
        integer site index.

    Raises:
        TypeError: If ``target_site_index`` is neither an integer nor ``None``.
    """
    sample_id = str(record["sample_id"])
    site_index = record.get("target_site_index")
    if site_index is not None and not isinstance(site_index, Integral):
        raise TypeError(f"target_site_index must be an integer or None, got {site_index!r}")
    return sample_id, None if site_index is None else int(site_index)


def sample_key_sort_key(key: SampleKey) -> tuple[str, int]:
    """Return a deterministic sort key for a compound sample identity.

    Args:
        key: Compound sample identity returned by :func:`sample_key`.

    Returns:
        ``(sample_id, site_index)`` with ``None`` site indices ordered first.
    """
    return key[0], -1 if key[1] is None else key[1]


def sample_label(record: Mapping[str, Any]) -> str:
    """Return a human-readable label that distinguishes target sites.

    Args:
        record: Prediction or collector record to label.

    Returns:
        Sample ID, optionally followed by its target-site index.
    """
    sample_id, site_index = sample_key(record)
    return f"{sample_id} (site {site_index})" if site_index is not None else sample_id


def is_scalar_value(value: Any) -> TypeGuard[ScalarValue]:
    """Return whether ``value`` is a non-boolean numeric scalar.

    Args:
        value: Candidate value from a prediction sample or collector output.

    Returns:
        ``True`` when ``value`` is a Python or NumPy integer/floating scalar,
        excluding booleans.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (np.integer, np.floating)):
        return True
    return False


def iter_scalar_items(
    record: Mapping[str, Any],
    include_metadata: bool = False,
) -> Iterator[tuple[str, ScalarValue]]:
    """Yield the measurable scalar entries of one prediction or collector record.

    Spectra arrays, structures, identifiers, and every other non-scalar or
    metadata entry are skipped, so all pipeline stages agree on which record
    entries count as scalar values.

    Args:
        record: Prediction sample or collector output mapping.
        include_metadata: Whether to also yield the scalar entries of
            :data:`SAMPLE_METADATA_KEYS`. Only per-sample views that describe
            a single sample should enable this; summaries over many samples must
            not treat identifiers as measurements.

    Yields:
        ``(key, value)`` pairs for every measurable scalar entry.
    """
    for key, value in record.items():
        if not include_metadata and key in SAMPLE_METADATA_KEYS:
            continue
        if is_scalar_value(value):
            yield key, value


def is_vector_value(value: Any) -> TypeGuard[VectorValue]:
    """Return whether ``value`` is a one-dimensional vector.

    Args:
        value: Candidate value from a prediction sample or collector output.

    Returns:
        ``True`` when ``value`` is a list, tuple, or one-dimensional NumPy
        array or torch tensor.
    """
    if isinstance(value, (list, tuple)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    if isinstance(value, torch.Tensor):
        return value.ndim == 1
    return False


def as_float_vector(value: VectorValue) -> np.ndarray:
    """Return ``value`` as a one-dimensional float array.

    Args:
        value: List, tuple, one-dimensional NumPy array, or one-dimensional
            torch tensor.

    Returns:
        The vector converted to a one-dimensional ``float`` NumPy array.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def iter_vector_items(
    record: Mapping[str, Any],
    include_metadata: bool = False,
) -> Iterator[tuple[str, VectorValue]]:
    """Yield the measurable vector entries of one prediction or collector record.

    Scalars, identifiers, the ``prediction`` and ``target`` spectra, and every
    other non-vector value are skipped, so the vector aggregator summarizes
    only vector measurements such as energy-resolved loss curves or per-sample
    encodings.

    Args:
        record: Prediction sample or collector output mapping.
        include_metadata: Whether to also yield the vector entries of
            :data:`SAMPLE_METADATA_KEYS`. Only per-sample views that describe a
            single sample should enable this; summaries over many samples must
            not treat identifiers as measurements.

    Yields:
        ``(key, value)`` pairs for every one-dimensional vector entry.
    """
    for key, value in record.items():
        if not include_metadata and key in SAMPLE_METADATA_KEYS:
            continue
        if key in SPECTRUM_KEYS:
            continue
        if is_vector_value(value):
            yield key, value
