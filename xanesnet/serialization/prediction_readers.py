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

"""Prediction reader classes for XANESNET inference outputs."""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import h5py
import numpy as np
import torch

from .prediction_writers import PredictionBatch

###############################################################################
############################### DATA STRUCTURE ################################
###############################################################################


class PredictionSample(TypedDict):
    """Single-absorber prediction record returned by ``PredictionReader``.

    All values are numpy arrays or torch tensors with no leading absorber
    dimension — each field corresponds to a single row of the equivalent
    ``PredictionBatch`` field.
    """

    # Required:
    prediction: np.ndarray | torch.Tensor
    target: np.ndarray | torch.Tensor

    # Optional:
    file_name: NotRequired[str]
    forward_time: NotRequired[float]
    forward_time_pass: NotRequired[float]


###############################################################################
################################# BASE CLASS ##################################
###############################################################################


class PredictionReader(ABC):
    """Abstract base class for reading saved XANESNET inference results.

    Implements the ``Iterator`` protocol so that callers can iterate over
    predictions one absorber at a time.

    Args:
        path: Directory (or file) containing the saved prediction data.
    """

    def __init__(self, path: str | Path):
        """Initialize a prediction reader for a saved predictions location.

        Args:
            path: Directory (or file container) holding persisted prediction
                outputs.
        """
        self.path = Path(path)
        self._length: int | None = None
        self._current_index: int = 0

        self._validate_path()

    @abstractmethod
    def _validate_path(self) -> None:
        """Validate that ``self.path`` contains valid prediction data.

        Raises:
            FileNotFoundError: If the path or expected files do not exist.
            ValueError: If the data at the path is not valid.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of absorber records in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> PredictionSample:
        """Return the ``PredictionSample`` at position ``index``.

        Args:
            index: Zero-based absorber index.

        Returns:
            A ``PredictionSample`` for the requested absorber.
        """
        ...

    def __iter__(self) -> Iterator[PredictionSample]:
        """Reset the iteration cursor and return ``self`` as the iterator.

        Returns:
            This reader instance, positioned at the first sample.
        """
        self._current_index = 0
        return self

    def __next__(self) -> PredictionSample:
        """Return the next ``PredictionSample`` and advance the cursor.

        Returns:
            The next absorber record.

        Raises:
            StopIteration: When all absorbers have been yielded.
        """
        if self._current_index >= len(self):
            raise StopIteration

        sample = self[self._current_index]
        self._current_index += 1
        return sample

    def get_all(self) -> PredictionBatch:
        """Load all predictions at once.

        Returns:
            A ``PredictionBatch`` with all absorbers stacked along the leading
            absorber dimension.
        """

        all_data: dict[str, list[Any]] = {}

        for sample in self:
            for key, value in sample.items():
                all_data.setdefault(key, []).append(value)

        # Reset iterator
        self._current_index = 0

        batch = {key: self._stack_values(values) for key, values in all_data.items()}
        return cast(PredictionBatch, batch)

    @staticmethod
    def _stack_values(values: list[Any]) -> np.ndarray:
        if not values:
            return np.array([])

        if all(isinstance(value, np.ndarray) for value in values):
            arrays = [value for value in values if isinstance(value, np.ndarray)]
            return np.stack(arrays, axis=0)

        return np.array(values)

    @staticmethod
    def _normalize_sample_value(value: Any) -> Any:
        """Normalize a single per-absorber value to a Python primitive or ndarray.

        - bytes → str
        - np.generic (e.g. np.float64, np.bool_) → Python primitive via .item()
        - 0-d ndarray (scalar stored as array) → Python primitive via .item()
        - 1-d+ string/bytes ndarray → error (not supported)
        - 1-d+ numeric ndarray → returned as-is
        """
        if isinstance(value, bytes):
            return value.decode("utf-8")

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            # 0-d arrays represent per-sample scalars → unwrap to Python primitive
            if value.ndim == 0:
                if value.dtype.kind in {"S", "U"}:
                    return value.astype("U").item()
                return value.item()

            # Multi-dimensional string/bytes arrays are not supported
            if value.dtype.kind in {"S", "U"}:
                raise ValueError("PredictionSample cannot contain string arrays")

            return value

        return value

    def close(self) -> None:
        """Close any open resources.  No-op by default; override in subclasses."""
        pass

    # for use in 'with' statements
    def __enter__(self) -> "PredictionReader":
        """Enter the context manager and return ``self``.

        Returns:
            This reader instance.
        """
        return self

    # for use in 'with' statements
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the context manager and close open resources.

        Args:
            exc_type: Exception type raised inside the context, if any.
            exc_val: Exception instance raised inside the context, if any.
            exc_tb: Traceback associated with the exception, if any.
        """
        self.close()


###############################################################################
################################# HDF5 CLASS ##################################
###############################################################################


class HDF5Reader(PredictionReader):
    """Prediction reader backed by an HDF5 file written by ``HDF5Writer``.

    Args:
        path: Directory containing the ``predictions.h5`` file.
    """

    def __init__(self, path: str | Path):
        """Open an HDF5-backed predictions directory.

        Args:
            path: Directory containing ``predictions.h5``.
        """
        self._h5: h5py.File | None = None
        self._group: h5py.Group | None = None
        super().__init__(path)

    def _validate_path(self) -> None:
        h5_file = self.path / "predictions.h5"
        if not h5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

        h5: h5py.File | None = None
        try:
            h5 = h5py.File(h5_file, "r")

            if "predictions" not in h5:
                raise ValueError(f"No 'predictions' group found in {h5_file}")

            group = h5["predictions"]

            if not isinstance(group, h5py.Group):
                raise TypeError(f"Expected Group, got {type(group).__name__}")

            self._h5 = h5
            self._group = group
        except Exception:
            if h5 is not None:
                h5.close()
            raise

    def __len__(self) -> int:
        """Return the number of absorber records stored in the HDF5 file.

        Returns:
            Number of persisted absorber rows.
        """
        if self._length is not None:
            return self._length

        if self._group is None:
            raise RuntimeError("Reader not properly initialized")

        # Get length from the first dataset
        for key in self._group.keys():
            dset = self._group[key]
            if isinstance(dset, h5py.Dataset):
                self._length = dset.shape[0]
                break

        if self._length is None:
            raise ValueError("No datasets found in predictions group")

        return self._length

    def __getitem__(self, index: int) -> PredictionSample:
        """Return a single absorber record from the HDF5 dataset.

        Args:
            index: Zero-based absorber index.

        Returns:
            Decoded prediction sample for ``index``.
        """
        if self._group is None:
            raise RuntimeError("Reader not properly initialized")

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        sample: dict[str, Any] = {}

        for key in self._group.keys():
            dset = self._group[key]
            if isinstance(dset, h5py.Dataset):
                data = dset[index]
                data = self._normalize_sample_value(data)

                sample[key] = data

        return cast(PredictionSample, sample)

    def get_all(self) -> PredictionBatch:
        """Load every HDF5 dataset into a single stacked prediction batch.

        Returns:
            Mapping from field names to full absorber-major numpy arrays.
        """
        if self._group is None:
            raise RuntimeError("Reader not properly initialized")

        batch: dict[str, np.ndarray] = {}

        for key in self._group.keys():
            dset = self._group[key]
            if isinstance(dset, h5py.Dataset):
                data = dset[:]
                # Variable-length string datasets return object arrays in h5py 3.x;
                # convert to proper Unicode dtype for consistency
                if h5py.check_string_dtype(dset.dtype) is not None:
                    data = data.astype("U")
                batch[key] = data

        return cast(PredictionBatch, batch)

    def close(self) -> None:
        """Close the underlying HDF5 file handle if it is still open."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._group = None


###############################################################################
################################# NUMPY CLASS #################################
###############################################################################


class NumpyReader(PredictionReader):
    """Prediction reader for ``.npz`` files written by ``NumpyWriter``.

    Expects files named ``sample_XXXXXX.npz`` in the given directory.

    Args:
        path: Directory containing the ``sample_*.npz`` files.
    """

    def __init__(self, path: str | Path):
        """Index a directory of per-sample ``.npz`` prediction files.

        Args:
            path: Directory containing ``sample_*.npz`` files.
        """
        self._sample_files: list[Path] = []
        super().__init__(path)

    def _validate_path(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: {self.path}")

        if not self.path.is_dir():
            raise ValueError(f"Path is not a directory: {self.path}")

        # Find all sample files and sort them
        self._sample_files = sorted(self.path.glob("sample_*.npz"))

        if not self._sample_files:
            raise FileNotFoundError(f"No sample_*.npz files found in {self.path}")

        logging.debug(f"Found {len(self._sample_files)} sample files in {self.path}")

    def __len__(self) -> int:
        """Return the number of discovered ``.npz`` sample files.

        Returns:
            Number of readable prediction samples.
        """
        return len(self._sample_files)

    def __getitem__(self, index: int) -> PredictionSample:
        """Load a single prediction sample from disk.

        Args:
            index: Zero-based absorber index.

        Returns:
            Decoded prediction sample for ``index``.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        sample_file = self._sample_files[index]

        with np.load(sample_file) as data:
            sample = {key: self._normalize_sample_value(data[key]) for key in data.files}

        return cast(PredictionSample, sample)


###############################################################################
################################# JSON CLASS ##################################
###############################################################################


class JSONReader(PredictionReader):
    """Prediction reader for ``.json`` files written by ``JSONWriter``.

    Expects files named ``sample_XXXXXX.json`` in the given directory.

    Args:
        path: Directory containing the ``sample_*.json`` files.
    """

    def __init__(self, path: str | Path):
        """Index a directory of per-sample JSON prediction files.

        Args:
            path: Directory containing ``sample_*.json`` files.
        """
        self._sample_files: list[Path] = []
        super().__init__(path)

    def _validate_path(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: {self.path}")

        if not self.path.is_dir():
            raise ValueError(f"Path is not a directory: {self.path}")

        # Find all sample files and sort them
        self._sample_files = sorted(self.path.glob("sample_*.json"))

        if not self._sample_files:
            raise FileNotFoundError(f"No sample_*.json files found in {self.path}")

        logging.debug(f"Found {len(self._sample_files)} sample files in {self.path}")

    def __len__(self) -> int:
        """Return the number of discovered JSON sample files.

        Returns:
            Number of readable prediction samples.
        """
        return len(self._sample_files)

    def __getitem__(self, index: int) -> PredictionSample:
        """Load a single prediction sample from a JSON file.

        Args:
            index: Zero-based absorber index.

        Returns:
            Decoded prediction sample for ``index``.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        sample_file = self._sample_files[index]

        with open(sample_file, "r") as f:
            data = json.load(f)

        sample: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, (bool, str, int, float)):
                sample[key] = value
            else:
                sample[key] = self._normalize_sample_value(np.array(value))

        return cast(PredictionSample, sample)


###############################################################################
############################ FORMAT DETECTION #################################
###############################################################################


def detect_prediction_format(path: str | Path) -> type[PredictionReader]:
    """Infer the prediction format from directory contents.

    Args:
        path: Path to the predictions directory.

    Returns:
        The appropriate ``PredictionReader`` subclass
        (``HDF5Reader``, ``NumpyReader``, or ``JSONReader``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no recognisable prediction files are found.
    """
    predictions_path = Path(path)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions path not found: {predictions_path}")

    # Try to detect format
    if (predictions_path / "predictions.h5").exists():
        return HDF5Reader
    elif list(predictions_path.glob("sample_*.npz")):
        return NumpyReader
    elif list(predictions_path.glob("sample_*.json")):
        return JSONReader
    else:
        raise ValueError(
            f"Could not detect prediction format in {predictions_path}. "
            f"Expected HDF5 (predictions.h5), Numpy (sample_*.npz), or JSON (sample_*.json) files."
        )
