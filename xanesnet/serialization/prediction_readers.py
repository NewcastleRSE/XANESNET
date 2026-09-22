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
from pymatgen.core import Molecule, Structure
from tqdm import tqdm

from xanesnet.datasources import DataSource

from .prediction_writers import PredictionBatch

###############################################################################
############################### DATA STRUCTURE ################################
###############################################################################


class PredictionSample(TypedDict):
    """Single-target-site prediction record returned by ``PredictionReader``.

    All values are numpy arrays or torch tensors with no leading target-site
    dimension - each field corresponds to a single row of the equivalent
    ``PredictionBatch`` field. ``prediction_std`` is present when inference
    produced an energy/channel-wise uncertainty estimate. ``sample_id`` is the
    identifier written by XANESNET inference. ``target_site_index`` identifies
    the original atom within that structure, or is ``None`` for non-site-specific
    predictions. ``time_per_spectrum`` is the forward-pass duration amortized
    over the spectra produced by the batch. This value is in seconds and covers
    only the model forward pass. Analysis may attach the matching raw
    ``structure`` from the inference datasource.
    """

    # Required:
    prediction: np.ndarray | torch.Tensor
    target: np.ndarray | torch.Tensor
    sample_id: str
    target_site_index: int | None

    # Optional:
    prediction_std: NotRequired[np.ndarray | torch.Tensor]
    time_per_spectrum: NotRequired[float]
    structure: NotRequired[Molecule | Structure]


###############################################################################
################################# BASE CLASS ##################################
###############################################################################


class PredictionReader(ABC):
    """Abstract base class for reading saved XANESNET inference results.

    Implements the ``Iterator`` protocol so that callers can iterate over
    predictions one target site at a time.

    Args:
        path: Directory (or file) containing the saved prediction data.
    """

    def __init__(self, path: str | Path):
        """Initialize a prediction reader for a saved predictions location."""
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
        """Return the total number of target-site records in the dataset.

        Returns:
            Number of target-site-level prediction records.
        """
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> PredictionSample:
        """Return the ``PredictionSample`` at position ``index``.

        Args:
            index: Zero-based target-site index.

        Returns:
            A ``PredictionSample`` for the requested target site.
        """
        ...

    def provides_structures(self) -> bool:
        """Return whether records carry matched raw structures.

        Returns:
            ``True`` when ``__getitem__`` results include a ``structure``
            entry, ``False`` otherwise.
        """
        return False

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
            The next target-site record.

        Raises:
            StopIteration: When all target sites have been yielded.
        """
        if self._current_index >= len(self):
            raise StopIteration

        sample = self[self._current_index]
        self._current_index += 1
        return sample

    def get_all(self) -> PredictionBatch:
        """Load all predictions at once.

        Returns:
            A ``PredictionBatch`` with all target sites stacked along the leading
            target-site dimension.
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
        """Stack prediction field values into one array."""
        if all(isinstance(value, np.ndarray) for value in values):
            arrays = [value for value in values if isinstance(value, np.ndarray)]
            return np.stack(arrays, axis=0)

        return np.array(values)

    @staticmethod
    def _normalize_sample_value(value: Any) -> Any:
        """Normalize a single per-target-site value to a Python primitive or ndarray.

        - bytes -> str
        - np.generic (e.g. np.float64, np.bool_) -> Python primitive via .item()
        - 0-d ndarray (scalar stored as array) -> Python primitive via .item()
        - 1-d+ string/bytes ndarray -> error (not supported)
        - 1-d+ numeric ndarray -> returned as-is
        """
        if isinstance(value, bytes):
            return value.decode("utf-8")

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            # 0-d arrays represent per-sample scalars -> unwrap to Python primitive
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
########################## STRUCTURE MATCHED CLASS ############################
###############################################################################


class StructureMatchedPredictionReader(PredictionReader):
    """Attach the structure matching each prediction's ``sample_id``.

    The wrapped prediction reader remains indexed at target-site level. A
    structure may therefore occur in multiple returned samples when it has
    multiple target sites. When ``preload_structures`` is enabled, parsed
    structures are retained in memory after indexing instead of being
    re-read from disk on every sample access.

    Args:
        predictions_reader: Reader containing persisted inference predictions.
        datasource: Raw datasource reconstructed from the inference run.
        preload_structures: Whether to cache parsed raw structures in memory.

    Raises:
        ValueError: If the datasource contains duplicate ``sample_id`` values,
            or a prediction refers to a sample that is absent from it.
    """

    def __init__(
        self,
        predictions_reader: PredictionReader,
        datasource: DataSource,
        preload_structures: bool = False,
    ) -> None:
        """Initialize the structure-enriched prediction reader.

        Args:
            predictions_reader: Reader containing persisted inference predictions.
            datasource: Raw datasource reconstructed from the inference run.
            preload_structures: Whether to cache parsed raw structures in memory.
        """
        self.predictions_reader = predictions_reader
        self.datasource = datasource
        self.preload_structures = preload_structures
        self._structure_cache: dict[int, Molecule | Structure] | None = {} if preload_structures else None
        self._source_indices_by_id = self._index_source_indices(datasource)
        super().__init__(predictions_reader.path)

    def _validate_path(self) -> None:
        """Accept the already validated path owned by the wrapped reader."""

    def __len__(self) -> int:
        """Return the number of target-site prediction records.

        Returns:
            Number of records provided by the wrapped prediction reader.
        """
        return len(self.predictions_reader)

    def provides_structures(self) -> bool:
        """Return whether records carry matched raw structures.

        Returns:
            Always ``True`` for this reader.
        """
        return True

    def __getitem__(self, index: int) -> PredictionSample:
        """Return one prediction record with its originating raw structure.

        Args:
            index: Zero-based target-site prediction index.

        Returns:
            Prediction record with an added ``structure`` entry.

        Raises:
            ValueError: If the record's ``sample_id`` is not in the inference
                datasource.
        """
        sample = dict(self.predictions_reader[index])
        sample_id = str(sample["sample_id"])
        try:
            source_index = self._source_indices_by_id[sample_id]
        except KeyError as exc:
            raise ValueError(f"Prediction at index {index} references unknown sample_id '{sample_id}'.") from exc
        sample["structure"] = self._load_structure(source_index)
        return cast(PredictionSample, sample)

    def get_all(self) -> PredictionBatch:
        """Load persisted prediction fields without materializing structures.

        Returns:
            Full prediction batch returned by the wrapped reader.
        """
        return self.predictions_reader.get_all()

    def close(self) -> None:
        """Close resources owned by the wrapped prediction reader."""
        self.predictions_reader.close()

    def _load_structure(self, source_index: int) -> Molecule | Structure:
        """Return the parsed raw structure for one datasource position.

        Cached entries are reused when ``preload_structures`` is enabled;
        otherwise the structure is loaded from disk on every access.

        Args:
            source_index: Zero-based datasource position.

        Returns:
            The pymatgen structure or molecule at ``source_index``.
        """
        if self._structure_cache is not None:
            return self._structure_cache[source_index]
        return self.datasource[source_index]

    def _index_source_indices(self, datasource: DataSource) -> dict[str, int]:
        """Index datasource positions by their required ``sample_id`` property.

        When ``preload_structures`` is enabled, the parsed objects are
        retained in ``_structure_cache`` for later reuse.

        Args:
            datasource: Source of raw structures or molecules.

        Returns:
            Mapping from sample identifier to datasource position.

        Raises:
            ValueError: If a structure lacks ``sample_id`` or the datasource
                contains duplicate identifiers.
        """
        source_indices_by_id: dict[str, int] = {}
        for source_index, structure in tqdm(
            enumerate(datasource), desc="Indexing datasource sample_ids", total=len(datasource)
        ):
            try:
                sample_id = str(structure.properties["sample_id"])
            except KeyError as exc:
                raise ValueError("Inference datasource entries must define properties['sample_id'].") from exc

            if sample_id in source_indices_by_id:
                raise ValueError(f"Inference datasource contains duplicate sample_id '{sample_id}'.")
            source_indices_by_id[sample_id] = source_index
            if self._structure_cache is not None:
                self._structure_cache[source_index] = structure

        return source_indices_by_id


###############################################################################
############################## PRELOAD CLASS ##################################
###############################################################################


class PreloadedPredictionReader(PredictionReader):
    """Serve an in-memory snapshot of a wrapped prediction reader's records.

    Materializes every target-site record of the wrapped reader once during
    setup and serves later accesses from memory. Useful when the analysis
    pipeline iterates the same predictions multiple times.

    Args:
        predictions_reader: Reader whose records should be preloaded.
    """

    def __init__(self, predictions_reader: PredictionReader) -> None:
        """Preload all records from the wrapped prediction reader."""
        self.predictions_reader = predictions_reader
        self._samples: list[PredictionSample] = [
            predictions_reader[i] for i in tqdm(range(len(predictions_reader)), desc="Preloading predictions")
        ]
        super().__init__(predictions_reader.path)

    def _validate_path(self) -> None:
        """Accept the already validated path owned by the wrapped reader."""

    def __len__(self) -> int:
        """Return the number of preloaded target-site records.

        Returns:
            Number of records served from memory.
        """
        return len(self._samples)

    def provides_structures(self) -> bool:
        """Forward the structure capability of the wrapped reader.

        Returns:
            ``True`` when the wrapped reader attaches matched raw structures.
        """
        return self.predictions_reader.provides_structures()

    def __getitem__(self, index: int) -> PredictionSample:
        """Return one preloaded prediction record.

        Args:
            index: Zero-based target-site prediction index.

        Returns:
            The preloaded ``PredictionSample`` at ``index``.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")
        return self._samples[index]

    def close(self) -> None:
        """Close resources owned by the wrapped prediction reader."""
        self.predictions_reader.close()


###############################################################################
################################# HDF5 CLASS ##################################
###############################################################################


class HDF5Reader(PredictionReader):
    """Prediction reader backed by an HDF5 file written by ``HDF5Writer``.

    Args:
        path: Directory containing the ``predictions.h5`` file.
    """

    def __init__(self, path: str | Path):
        """Open an HDF5-backed predictions directory."""
        self._h5: h5py.File | None = None
        self._group: h5py.Group | None = None
        super().__init__(path)

    def _validate_path(self) -> None:
        """Validate the prediction storage path."""
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
        """Return the number of target-site records stored in the HDF5 file.

        Returns:
            Number of persisted target-site rows.
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
        """Return a single target-site record from the HDF5 dataset.

        Args:
            index: Zero-based target-site index.

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

        sample.setdefault("target_site_index", None)
        return cast(PredictionSample, sample)

    def get_all(self) -> PredictionBatch:
        """Load every HDF5 dataset into a single stacked prediction batch.

        Returns:
            Mapping from field names to full target-site-major numpy arrays.
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
        """Index a directory of per-sample ``.npz`` prediction files."""
        self._sample_files: list[Path] = []
        super().__init__(path)

    def _validate_path(self) -> None:
        """Validate the prediction storage path."""
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
            index: Zero-based target-site index.

        Returns:
            Decoded prediction sample for ``index``.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        sample_file = self._sample_files[index]

        with np.load(sample_file) as data:
            sample = {key: self._normalize_sample_value(data[key]) for key in data.files}

        sample.setdefault("target_site_index", None)
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
        """Index a directory of per-sample JSON prediction files."""
        self._sample_files: list[Path] = []
        super().__init__(path)

    def _validate_path(self) -> None:
        """Validate the prediction storage path."""
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
            index: Zero-based target-site index.

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

        sample.setdefault("target_site_index", None)
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
        ValueError: If no recognizable prediction files are found.
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


###############################################################################
################################# FACTORY #####################################
###############################################################################


def build_prediction_reader(
    path: str | Path,
    datasource: DataSource | None = None,
    preload: bool = False,
) -> PredictionReader:
    """Build a prediction reader for one predictions directory.

    Detects the storage format from the directory contents, optionally
    attaches matching raw structures from ``datasource``, and optionally
    preloads every target-site record into memory.

    Args:
        path: Directory containing persisted prediction files.
        datasource: Optional datasource whose raw structures are attached to
            matching prediction records. When ``None``, records are returned
            without structures.
        preload: Whether to preload prediction records and matched structures
            into memory during setup.

    Returns:
        A format reader, optionally wrapped for structure enrichment and
        in-memory preloading.
    """
    reader_class = detect_prediction_format(path)
    logging.info(f"Detected format for {path}: {reader_class.__name__}")
    reader = reader_class(path)

    if datasource is not None:
        reader = StructureMatchedPredictionReader(reader, datasource, preload_structures=preload)
    if preload:
        reader = PreloadedPredictionReader(reader)

    return reader
