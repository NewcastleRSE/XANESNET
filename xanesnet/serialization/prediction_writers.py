"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
    """
    PredictionBatch to be saved by PredictionWriter.

    Everything is provided as numpy arrays or torch tensors.
    First dimension is batch dimension. All array-likes must have the same leading batch size.
    """

    # Required:
    prediction: np.ndarray | torch.Tensor
    target: np.ndarray | torch.Tensor

    # Optional:
    input: NotRequired[dict[str, np.ndarray | torch.Tensor]]
    file_name: NotRequired[np.ndarray | torch.Tensor]
    forward_time: NotRequired[np.ndarray | torch.Tensor]


###############################################################################
################################# BASE CLASS ##################################
###############################################################################


class PredictionWriter(ABC):
    """
    Abstract base class for writers.
    """

    def __init__(self, path: str | Path, buffer_size: int):
        self.path = Path(path)
        self.buffer_size = buffer_size

        self._buffers: dict[str, list[np.ndarray]] = {}
        self._buffer_count: int = 0
        self._total_written: int = 0

        self._init_storage()

    def add(self, batch: PredictionBatch) -> None:
        """
        Add a PredictionBatch to the buffer.
        """
        batch_size: int | None = None

        for key, value in batch.items():
            array = self._to_numpy(value)

            if array.ndim == 0:
                raise ValueError(f"Value for key '{key}' is scalar; expected batch dimension")

            # Bool and string are only supported as per-sample scalars (1D in batch)
            if array.dtype.kind in ("U", "S", "b") and array.ndim > 1:
                raise TypeError(
                    f"Key '{key}': {array.dtype} arrays are not supported, "
                    f"only per-sample scalars (got shape {array.shape})"
                )

            if batch_size is None:
                batch_size = array.shape[0]
            elif array.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch for key '{key}': expected {batch_size}, got {array.shape[0]}")

            self._buffers.setdefault(key, []).append(array)

        if batch_size is None:
            raise ValueError("Empty PredictionBatch provided")

        self._buffer_count += batch_size

        if self._buffer_count >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """
        Flush buffered data to storage.
        """
        if self._buffer_count == 0:
            logging.debug("No data to flush.")
            return

        batch = {key: np.concatenate(chunks, axis=0) for key, chunks in self._buffers.items()}

        self._write_batch(batch)

        self._total_written += self._buffer_count
        self._buffers.clear()
        self._buffer_count = 0

    def close(self) -> None:
        """
        Flush remaining data and close storage.
        """
        self.flush()
        self._close_storage()

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        raise ValueError(f"Unsupported type: {type(x)}")

    def _init_storage(self) -> None:
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
                    "  - NumpyWriter: Stores one .npz file per sample (good for debugging).\n"
                    "  - JSONWriter: Stores one .json file per sample (human readable).\n"
                )

    def _close_storage(self) -> None:
        pass  # Nothing to close by default

    @abstractmethod
    def _write_batch(self, batch: dict[str, np.ndarray]) -> None: ...


###############################################################################
################################# HDF5 CLASS ##################################
###############################################################################


class HDF5Writer(PredictionWriter):
    """
    HDF5-backed inference writer.

    Supported per-sample data:
        - float / int arrays of any shape
        - scalar float, int, bool, or string values

    Bool and string *arrays* (ndim > 0 per sample) are not supported.
    """

    def __init__(
        self,
        path: str | Path,
        buffer_size: int = 100_000,
        compression: str = "gzip",
    ):
        self.compression = compression
        super().__init__(path, buffer_size)

    def _init_storage(self) -> None:
        super()._init_storage()

        self._h5: h5py.File = h5py.File(self.path / "predictions.h5", "w")
        self._group: h5py.Group = self._h5.create_group("predictions")

    def _ensure_dataset(self, key: str, data: np.ndarray) -> None:
        if key in self._group:
            return

        shape = (0,) + data.shape[1:]
        maxshape = (None,) + data.shape[1:]

        dtype = data.dtype
        compression = self.compression

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
        self._h5.close()


class NumpyWriter(PredictionWriter):
    """
    Writes one .npz file per sample containing all arrays.

    Mostly useful for debugging or small datasets.
    """

    def _write_batch(self, batch: dict[str, np.ndarray]) -> None:
        batch_size = next(iter(batch.values())).shape[0]

        for i in range(batch_size):
            sample_file = self.path / f"sample_{self._total_written + i:06d}.npz"
            sample_data = {key: data[i] for key, data in batch.items()}
            np.savez(sample_file, **sample_data)


class JSONWriter(PredictionWriter):
    """
    Writes one JSON file per sample.

    Mostly useful for debugging or small datasets.
    """

    def _write_batch(self, batch: dict[str, np.ndarray]) -> None:
        batch_size = next(iter(batch.values())).shape[0]

        for i in range(batch_size):
            sample_data = {key: data[i].tolist() for key, data in batch.items()}
            sample_file = self.path / f"sample_{self._total_written + i:06d}.json"

            with open(sample_file, "w") as f:
                json.dump(sample_data, f, indent=2)
