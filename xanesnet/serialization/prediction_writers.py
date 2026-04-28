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

"""Prediction writer classes for XANESNET inference outputs."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import h5py
import numpy as np
import torch

###############################################################################
############################### DATA STRUCTURE ################################
###############################################################################


class PredictionBatch(TypedDict):
    """Batch of per-absorber predictions ready to be persisted.

    All values are numpy arrays or torch tensors whose first dimension is the
    absorber dimension - every field carries one row per absorber site.  All
    array-like fields must share the same leading absorber size.
    """

    # Required:
    prediction: np.ndarray | torch.Tensor
    target: np.ndarray | torch.Tensor

    # Optional:
    file_name: NotRequired[np.ndarray]
    forward_time: NotRequired[np.ndarray | torch.Tensor]
    forward_time_pass: NotRequired[np.ndarray | torch.Tensor]


###############################################################################
################################# BASE CLASS ##################################
###############################################################################


class PredictionWriter(ABC):
    """Abstract base class for writing XANESNET inference results to disk.

    Buffers incoming ``PredictionBatch`` objects and flushes them in batches
    to keep memory usage bounded.  All storage is indexed along the absorber
    dimension.

    Args:
        path: Directory where output data will be written.
        buffer_size: Number of absorber rows to accumulate before flushing to
            storage.
    """

    def __init__(self, path: str | Path, buffer_size: int):
        """Initialize a prediction writer and its on-disk storage."""
        self.path = Path(path)
        self.buffer_size = buffer_size

        self._buffers: dict[str, list[np.ndarray]] = {}
        self._buffer_count: int = 0
        self._total_written: int = 0
        self._expected_keys: set[str] | None = None

        self._init_storage()

    def add(self, batch: PredictionBatch) -> None:
        """Buffer a batch of absorber predictions.

        The batch is accumulated in an internal buffer.  When the buffer
        reaches ``buffer_size`` rows, it is automatically flushed to storage.

        Args:
            batch: A ``PredictionBatch`` where every field's first dimension is
                the absorber dimension.  All fields must have the same leading
                absorber size.

        Raises:
            ValueError: If any value has no leading absorber dimension, the
                absorber sizes across fields are inconsistent, the set of
                fields differs from earlier batches, or the batch is empty.
            TypeError: If a boolean or string array has more than one dimension
                per absorber.
        """
        batch_keys = set(batch)
        if self._expected_keys is None:
            self._expected_keys = batch_keys
        elif batch_keys != self._expected_keys:
            missing = sorted(self._expected_keys - batch_keys)
            extra = sorted(batch_keys - self._expected_keys)
            details: list[str] = []
            if missing:
                details.append(f"missing keys: {missing}")
            if extra:
                details.append(f"unexpected keys: {extra}")
            raise ValueError(
                "PredictionBatch keys must stay consistent across writes; " + "; ".join(details)
            )

        n_absorbers: int | None = None

        for key, value in batch.items():
            array = self._to_numpy(value)

            if array.ndim == 0:
                raise ValueError(f"Value for key '{key}' is scalar; expected leading absorber dimension")

            # Bool and string are only supported as per-absorber scalars
            # (1-D along the absorber dimension).
            if array.dtype.kind in ("U", "S", "b") and array.ndim > 1:
                raise TypeError(
                    f"Key '{key}': {array.dtype} arrays are not supported, "
                    f"only per-absorber scalars (got shape {array.shape})"
                )

            if n_absorbers is None:
                n_absorbers = array.shape[0]
            elif array.shape[0] != n_absorbers:
                raise ValueError(
                    f"Absorber-dimension mismatch for key '{key}': expected {n_absorbers}, got {array.shape[0]}"
                )

            self._buffers.setdefault(key, []).append(array)

        if n_absorbers is None:
            raise ValueError("Empty PredictionBatch provided")

        self._buffer_count += n_absorbers

        if self._buffer_count >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush all buffered data to storage.

        Concatenates buffered chunks for each field and delegates to
        ``_write_batch``.  Clears the buffer afterwards.
        """
        if self._buffer_count == 0:
            logging.debug("No data to flush.")
            return

        flushed = {key: np.concatenate(chunks, axis=0) for key, chunks in self._buffers.items()}

        self._write_batch(flushed)

        self._total_written += self._buffer_count
        self._buffers.clear()
        self._buffer_count = 0

    def close(self) -> None:
        """Flush any remaining buffered data and close storage resources."""
        self.flush()
        self._close_storage()

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        """Convert a value to a numpy array.

        Args:
            x: A ``torch.Tensor`` or ``np.ndarray``.

        Returns:
            The input as a detached, CPU-resident ``np.ndarray``.

        Raises:
            ValueError: If ``x`` is neither a tensor nor a numpy array.
        """
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        raise ValueError(f"Unsupported type: {type(x)}")

    def _init_storage(self) -> None:
        """Initialize output directory and write the ``WRITER_INFO.txt`` descriptor.

        Called once during ``__init__``.  Subclasses should call
        ``super()._init_storage()`` and then set up format-specific resources.
        """
        self.path.mkdir(parents=True, exist_ok=True)  # TODO not sure if needed

        info_file = self.path / "WRITER_INFO.txt"
        if not info_file.exists():
            with open(info_file, "w") as f:
                f.write(
                    "XANESNET Prediction Output\n"
                    "==========================\n\n"
                    f"This data was generated using: {self.__class__.__name__}\n\n"
                    "You can configure the writer type by changing the code in the inferencer.\n"
                    "Available writers:\n"
                    "  - HDF5Writer (default): Stores all predictions in a single HDF5 file.\n"
                    "  - NumpyWriter: Stores one .npz file per absorber (good for debugging).\n"
                    "  - JSONWriter: Stores one .json file per absorber (human readable).\n"
                )

    def _close_storage(self) -> None:
        """Release format-specific storage resources.  No-op by default."""
        pass  # Nothing to close by default

    @abstractmethod
    def _write_batch(self, batch: dict[str, np.ndarray]) -> None:
        """Persist a fully concatenated batch of absorber rows.

        Args:
            batch: Mapping from field name to a numpy array whose first
                dimension is the absorber dimension.
        """
        ...


###############################################################################
################################# HDF5 CLASS ##################################
###############################################################################


class HDF5Writer(PredictionWriter):
    """HDF5-backed prediction writer.

    Appends absorber rows to datasets inside a single ``predictions.h5`` file.

    Supported per-absorber payload types:
    - Numeric arrays of any shape.
    - Per-absorber scalar ``float``, ``int``, or ``bool`` values.
    - Per-absorber scalar ``str`` / ``bytes`` values.

    Bool and string *arrays* (``ndim > 0`` per absorber) are not supported.

    Args:
        path: Directory in which ``predictions.h5`` will be created.
        buffer_size: Number of absorber rows to buffer before flushing.
        compression: HDF5 compression filter name (default ``"gzip"``).
    """

    def __init__(
        self,
        path: str | Path,
        buffer_size: int = 100_000,
        compression: str = "gzip",
    ):
        """Initialize an HDF5-backed prediction writer."""
        self.compression = compression
        super().__init__(path, buffer_size)

    def _init_storage(self) -> None:
        """Create the output directory and open the HDF5 file for writing."""
        super()._init_storage()

        self._h5: h5py.File = h5py.File(self.path / "predictions.h5", "w")
        self._group: h5py.Group = self._h5.create_group("predictions")

    def _ensure_dataset(self, key: str, data: np.ndarray) -> None:
        """Create an HDF5 dataset for ``key`` if one does not already exist.

        Args:
            key: Dataset name (field name from the prediction batch).
            data: A representative slice used to infer dtype and shape.
        """
        if key in self._group:
            return

        shape = (0,) + data.shape[1:]
        maxshape = (None,) + data.shape[1:]

        dtype = data.dtype
        compression: str | None = self.compression

        if dtype.kind == "U":
            dtype = h5py.string_dtype(encoding="utf-8", length=None)
            # HDF5 does not support filters on variable-length types
            compression = None

        self._group.create_dataset(
            key,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=True,
            compression=compression,
        )

    def _write_batch(self, batch: dict[str, np.ndarray]) -> None:
        """Append a batch of absorber rows to the HDF5 file.

        Args:
            batch: Mapping from field name to concatenated numpy arrays.
        """
        for key, data in batch.items():
            self._ensure_dataset(key, data)

            dset = self._group[key]
            # Type narrowing: ensure we're working with a Dataset
            if not isinstance(dset, h5py.Dataset):
                raise TypeError(f"Expected Dataset, got {type(dset).__name__}")

            start = dset.shape[0]
            dset.resize(start + data.shape[0], axis=0)
            dset[start : start + data.shape[0]] = data

    def _close_storage(self) -> None:
        """Flush and close the HDF5 file."""
        self._h5.close()


class NumpyWriter(PredictionWriter):
    """Prediction writer that saves one ``.npz`` file per absorber.

    Each file is named ``sample_XXXXXX.npz`` and contains all fields for that
    absorber.  Useful for debugging or small datasets.
    """

    def _write_batch(self, batch: dict[str, np.ndarray]) -> None:
        """Write one prediction batch to disk."""
        n_absorbers = next(iter(batch.values())).shape[0]

        for i in range(n_absorbers):
            sample_file = self.path / f"sample_{self._total_written + i:06d}.npz"
            sample_data = {key: data[i] for key, data in batch.items()}
            np.savez(sample_file, **sample_data)


class JSONWriter(PredictionWriter):
    """Prediction writer that saves one ``.json`` file per absorber.

    Each file is named ``sample_XXXXXX.json`` and contains all fields for that
    absorber as human-readable JSON.  Useful for debugging or small datasets.
    """

    def _write_batch(self, batch: dict[str, np.ndarray]) -> None:
        """Write one prediction batch to disk."""
        n_absorbers = next(iter(batch.values())).shape[0]

        for i in range(n_absorbers):
            sample_data = {key: data[i].tolist() for key, data in batch.items()}
            sample_file = self.path / f"sample_{self._total_written + i:06d}.json"

            with open(sample_file, "w") as f:
                json.dump(sample_data, f, indent=2)
