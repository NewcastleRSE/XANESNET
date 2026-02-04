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
from typing import Any, Iterator

import h5py
import numpy as np

###############################################################################
################################# BASE CLASS ##################################
###############################################################################


class PredictionReader(ABC):
    """
    Abstract base class for prediction readers.

    Implements the Iterator protocol for iterating over saved predictions.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._length: int | None = None
        self._current_index: int = 0

        self._validate_path()

    @abstractmethod
    def _validate_path(self) -> None:
        """
        Validate that the path contains valid prediction data.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples.
        """
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """
        Get a single sample by index.
        """
        ...

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """
        Return iterator over all samples.
        """
        self._current_index = 0
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        """
        Get the next sample.
        """
        if self._current_index >= len(self):
            raise StopIteration

        sample = self[self._current_index]
        self._current_index += 1
        return sample

    def get_all(self) -> dict[str, np.ndarray]:
        """
        Load all predictions at once.

        Returns a dictionary with all samples stacked along the batch dimension.
        """
        all_data: dict[str, list[Any]] = {}

        for sample in self:
            for key, value in sample.items():
                all_data.setdefault(key, []).append(value)

        # Reset iterator
        self._current_index = 0

        return {key: np.stack(arrays, axis=0) for key, arrays in all_data.items()}

    def close(self) -> None:
        """
        Close any open resources. Override in subclasses if needed.
        """
        pass

    # for use in 'with' statements
    def __enter__(self) -> "PredictionReader":
        return self

    # for use in 'with' statements
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


###############################################################################
################################# HDF5 CLASS ##################################
###############################################################################


class HDF5Reader(PredictionReader):
    """
    HDF5-backed prediction reader.

    Reads predictions saved by HDF5Writer.
    """

    def __init__(self, path: str | Path):
        self._h5: h5py.File | None = None
        self._group: h5py.Group | None = None
        super().__init__(path)

    def _validate_path(self) -> None:
        h5_file = self.path / "predictions.h5"
        if not h5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

        self._h5 = h5py.File(h5_file, "r")

        if "predictions" not in self._h5:
            raise ValueError(f"No 'predictions' group found in {h5_file}")

        group = self._h5["predictions"]

        if not isinstance(group, h5py.Group):
            raise TypeError(f"Expected Group, got {type(group).__name__}")

        self._group = group

    def __len__(self) -> int:
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

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if self._group is None:
            raise RuntimeError("Reader not properly initialized")

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        sample: dict[str, np.ndarray] = {}

        for key in self._group.keys():
            dset = self._group[key]
            if isinstance(dset, h5py.Dataset):
                data = dset[index]
                # Convert bytes to string if needed
                if isinstance(data, bytes):
                    data = np.array(data.decode("utf-8"))
                elif isinstance(data, np.ndarray) and data.dtype.kind == "S":
                    data = data.astype("U")
                elif not isinstance(data, np.ndarray):
                    data = np.array(data)

                sample[key] = data

        return sample

    def get_all(self) -> dict[str, np.ndarray]:
        if self._group is None:
            raise RuntimeError("Reader not properly initialized")

        sample: dict[str, np.ndarray] = {}

        for key in self._group.keys():
            dset = self._group[key]
            if isinstance(dset, h5py.Dataset):
                data = dset[:]
                # Convert bytes to string if needed
                if data.dtype.kind == "S":
                    data = data.astype("U")
                sample[key] = data

        return sample

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._group = None


###############################################################################
################################# NUMPY CLASS #################################
###############################################################################


class NumpyReader(PredictionReader):
    """
    Reads .npz files saved by NumpyWriter.

    Expects files named sample_XXXXXX.npz in the given directory.
    """

    def __init__(self, path: str | Path):
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
        return len(self._sample_files)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        sample_file = self._sample_files[index]

        with np.load(sample_file) as data:
            sample: dict[str, np.ndarray] = {key: data[key] for key in data.files}

        return sample


###############################################################################
################################# JSON CLASS ##################################
###############################################################################


class JSONReader(PredictionReader):
    """
    Reads JSON files saved by JSONWriter.

    Expects files named sample_XXXXXX.json in the given directory.
    """

    def __init__(self, path: str | Path):
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
        return len(self._sample_files)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        sample_file = self._sample_files[index]

        with open(sample_file, "r") as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        sample: dict[str, np.ndarray] = {key: np.array(value) for key, value in data.items()}

        return sample
