# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Unit tests for persisting predictions (writers) and reading them back (readers)."""

import json
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest
import torch
from pymatgen.core import Molecule

from xanesnet.datasources import DataSource
from xanesnet.serialization.prediction_readers import (
    HDF5Reader,
    JSONReader,
    NumpyReader,
    PredictionReader,
    PreloadedPredictionReader,
    StructureMatchedPredictionReader,
    build_prediction_reader,
    detect_prediction_format,
)
from xanesnet.serialization.prediction_writers import (
    HDF5Writer,
    JSONWriter,
    NumpyWriter,
    PredictionBatch,
)

WRITERS = [HDF5Writer, NumpyWriter, JSONWriter]
WRITER_READER_PAIRS = [(HDF5Writer, HDF5Reader), (NumpyWriter, NumpyReader), (JSONWriter, JSONReader)]
PAIR_IDS = ["hdf5", "numpy", "json"]

N_ENERGIES = 5


def make_batch(
    n: int = 4,
    offset: int = 0,
    *,
    prediction_std: np.ndarray | torch.Tensor | None = None,
    time_per_spectrum: np.ndarray | torch.Tensor | None = None,
) -> PredictionBatch:
    """Build a batch whose target-site values encode their global row numbers.

    Args:
        n: Number of target sites in the batch.
        offset: Global row number of the first target site.
        prediction_std: Optional uncertainty values for each target site.
        time_per_spectrum: Optional model-forward duration for each target site.

    Returns:
        Batch mapping containing prediction, target, sample, and site data.
    """
    rows = np.arange(offset, offset + n)
    batch: PredictionBatch = {
        "prediction": (rows[:, None] + np.linspace(0, 1, N_ENERGIES)[None, :]).astype(np.float32),
        "target": (rows[:, None] * 2.0 + np.linspace(0, 1, N_ENERGIES)[None, :]).astype(np.float32),
        "sample_id": np.array([f"sample_{r}" for r in rows]),
        "target_site_index": rows % 3,
    }
    if prediction_std is not None:
        batch["prediction_std"] = prediction_std
    if time_per_spectrum is not None:
        batch["time_per_spectrum"] = time_per_spectrum
    return batch


class ListDataSource(DataSource):
    """Minimal in-memory datasource of molecules."""

    def __init__(self, molecules: list[Molecule]) -> None:
        super().__init__("test")
        self.molecules = molecules

    def __iter__(self):
        return iter(self.molecules)

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Molecule:
        return self.molecules[idx]


def make_molecule(sample_id: str | None) -> Molecule:
    """Create a two-hydrogen molecule with an optional sample identifier.

    Args:
        sample_id: Identifier to store on the molecule, or ``None`` to omit it.

    Returns:
        Molecule with the requested sample properties.
    """
    properties = {} if sample_id is None else {"sample_id": sample_id}
    return Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]], properties=properties)


###############################################################################
############################### WRITER CONTRACT ###############################
###############################################################################


@pytest.mark.parametrize("writer_cls", WRITERS)
def test_writer_creates_directory_and_descriptor_file(writer_cls, tmp_path):
    writer = writer_cls(tmp_path / "nested" / "predictions", buffer_size=10)
    writer.close()

    info = (tmp_path / "nested" / "predictions" / "WRITER_INFO.txt").read_text()
    assert writer_cls.__name__ in info


@pytest.mark.parametrize("writer_cls, suffix", [(NumpyWriter, "npz"), (JSONWriter, "json")])
def test_rows_stay_buffered_until_the_buffer_is_full(writer_cls, suffix, tmp_path):
    """File-per-row writers make the flush behaviour observable through the files on disk."""
    writer = writer_cls(tmp_path, buffer_size=10)

    writer.add(make_batch(3))
    assert list(tmp_path.glob(f"sample_*.{suffix}")) == []

    writer.add(make_batch(7, offset=3))  # reaches buffer_size -> automatic flush
    assert len(list(tmp_path.glob(f"sample_*.{suffix}"))) == 10
    writer.close()


@pytest.mark.parametrize("writer_cls, suffix", [(NumpyWriter, "npz"), (JSONWriter, "json")])
def test_explicit_flush_writes_partial_buffers(writer_cls, suffix, tmp_path):
    writer = writer_cls(tmp_path, buffer_size=100)
    writer.add(make_batch(3))
    writer.flush()
    assert len(list(tmp_path.glob(f"sample_*.{suffix}"))) == 3
    writer.close()


@pytest.mark.parametrize("writer_cls, reader_cls", WRITER_READER_PAIRS, ids=PAIR_IDS)
def test_close_flushes_the_remaining_rows(writer_cls, reader_cls, tmp_path):
    writer = writer_cls(tmp_path, buffer_size=100)
    writer.add(make_batch(3))
    writer.close()

    with reader_cls(tmp_path) as reader:
        assert len(reader) == 3


@pytest.mark.parametrize("writer_cls", WRITERS)
def test_flushing_an_empty_buffer_is_harmless(writer_cls, tmp_path):
    writer = writer_cls(tmp_path, buffer_size=5)
    writer.flush()
    writer.close()


@pytest.mark.parametrize("writer_cls, reader_cls", WRITER_READER_PAIRS, ids=PAIR_IDS)
def test_torch_tensors_are_accepted(writer_cls, reader_cls, tmp_path):
    writer = writer_cls(tmp_path, buffer_size=5)
    batch = make_batch(2)
    batch["prediction"] = torch.from_numpy(batch["prediction"]).requires_grad_(True)
    writer.add(batch)
    writer.close()

    with reader_cls(tmp_path) as reader:
        np.testing.assert_allclose(reader[1]["prediction"], make_batch(2)["prediction"][1], rtol=1e-6)


@pytest.mark.parametrize("writer_cls", WRITERS)
class TestWriterValidation:
    def test_scalar_values_are_rejected(self, writer_cls, tmp_path):
        writer = writer_cls(tmp_path, buffer_size=5)
        batch = make_batch(2)
        batch["prediction"] = np.array(1.0, dtype=np.float32)
        with pytest.raises(ValueError, match="scalar; expected leading target-site dimension"):
            writer.add(batch)
        writer.close()

    def test_mismatched_target_site_counts_are_rejected(self, writer_cls, tmp_path):
        writer = writer_cls(tmp_path, buffer_size=5)
        batch = make_batch(3)
        batch["target"] = batch["target"][:2]
        with pytest.raises(ValueError, match="Target-site-dimension mismatch for key 'target'"):
            writer.add(batch)
        writer.close()

    def test_changing_the_field_set_between_batches_is_rejected(self, writer_cls, tmp_path):
        writer = writer_cls(tmp_path, buffer_size=50)
        writer.add(make_batch(2))
        with pytest.raises(ValueError, match="keys must stay consistent") as excinfo:
            writer.add(make_batch(2, prediction_std=np.zeros((2, N_ENERGIES), dtype=np.float32)))
        assert "prediction_std" in str(excinfo.value)
        writer.close()

    def test_missing_field_is_reported(self, writer_cls, tmp_path):
        writer = writer_cls(tmp_path, buffer_size=50)
        writer.add(make_batch(2))
        incomplete = cast(dict[str, Any], make_batch(2))
        del incomplete["target"]
        with pytest.raises(ValueError, match="missing keys: \\['target'\\]"):
            writer.add(cast(PredictionBatch, incomplete))
        writer.close()

    def test_multidimensional_string_arrays_are_rejected(self, writer_cls, tmp_path):
        writer = writer_cls(tmp_path, buffer_size=5)
        batch = cast(dict[str, Any], make_batch(2))
        batch["labels"] = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(TypeError, match="not supported"):
            writer.add(cast(PredictionBatch, batch))
        writer.close()

    def test_unsupported_value_types_are_rejected(self, writer_cls, tmp_path):
        writer = writer_cls(tmp_path, buffer_size=5)
        batch = cast(dict[str, Any], make_batch(2))
        batch["prediction"] = [[1.0], [2.0]]
        with pytest.raises(ValueError, match="Unsupported type"):
            writer.add(cast(PredictionBatch, batch))
        writer.close()

    def test_empty_batches_are_rejected(self, writer_cls, tmp_path):
        writer = writer_cls(tmp_path, buffer_size=5)
        with pytest.raises(ValueError, match="Empty PredictionBatch"):
            writer.add(cast(PredictionBatch, {}))
        writer.close()


###############################################################################
################################ FILE LAYOUT ###################################
###############################################################################


def test_hdf5_writer_stores_one_dataset_per_field_under_predictions(tmp_path):
    writer = HDF5Writer(tmp_path, buffer_size=3)
    writer.add(make_batch(5))
    writer.close()

    with h5py.File(tmp_path / "predictions.h5", "r") as h5:
        predictions = h5["predictions"]
        assert isinstance(predictions, h5py.Group)
        assert set(predictions) == {"prediction", "target", "sample_id", "target_site_index"}
        prediction = predictions["prediction"]
        assert isinstance(prediction, h5py.Dataset)
        assert prediction.shape == (5, N_ENERGIES)


def test_hdf5_writer_appends_across_flushes_in_order(tmp_path):
    writer = HDF5Writer(tmp_path, buffer_size=2)
    for start in (0, 2, 4):
        writer.add(make_batch(2, offset=start))
    writer.close()

    with h5py.File(tmp_path / "predictions.h5", "r") as h5:
        target_site_index = h5["predictions/target_site_index"]
        assert isinstance(target_site_index, h5py.Dataset)
        np.testing.assert_array_equal(target_site_index[:], np.arange(6) % 3)


def test_numpy_and_json_writers_create_one_numbered_file_per_row(tmp_path):
    for writer_cls, suffix in [(NumpyWriter, "npz"), (JSONWriter, "json")]:
        directory = tmp_path / suffix
        writer = writer_cls(directory, buffer_size=2)
        writer.add(make_batch(3))
        writer.add(make_batch(2, offset=3))
        writer.close()

        assert sorted(p.name for p in directory.glob(f"sample_*.{suffix}")) == [
            f"sample_{i:06d}.{suffix}" for i in range(5)
        ]


def test_json_writer_files_are_human_readable_json(tmp_path):
    writer = JSONWriter(tmp_path, buffer_size=1)
    writer.add(make_batch(1))
    writer.close()

    content = json.loads((tmp_path / "sample_000000.json").read_text())
    assert content["sample_id"] == "sample_0"
    assert len(content["prediction"]) == N_ENERGIES


###############################################################################
################################### ROUND TRIP ################################
###############################################################################


@pytest.fixture(params=WRITER_READER_PAIRS, ids=PAIR_IDS)
def written(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[type[PredictionReader], Path, PredictionBatch]:
    """Write 7 rows (with a mid-way flush) and return ``(reader_cls, path, batch)``."""
    writer_cls, reader_cls = request.param
    reader_cls: type[PredictionReader] = reader_cls
    batch = make_batch(7, time_per_spectrum=np.linspace(0.1, 0.7, 7).astype(np.float32))
    writer = writer_cls(tmp_path, buffer_size=4)
    batch_arrays = cast(dict[str, np.ndarray], batch)
    writer.add(cast(PredictionBatch, {key: value[:5] for key, value in batch_arrays.items()}))
    writer.add(cast(PredictionBatch, {key: value[5:] for key, value in batch_arrays.items()}))
    writer.close()
    return reader_cls, tmp_path, batch


def test_reader_reports_the_number_of_rows(written):
    reader_cls, path, batch = written
    with reader_cls(path) as reader:
        assert len(reader) == 7


def test_samples_round_trip_values_and_python_types(written):
    reader_cls, path, batch = written
    with reader_cls(path) as reader:
        sample = reader[3]

    np.testing.assert_allclose(sample["prediction"], batch["prediction"][3], rtol=1e-6)
    np.testing.assert_allclose(sample["target"], batch["target"][3], rtol=1e-6)
    assert sample["sample_id"] == "sample_3"
    assert isinstance(sample["sample_id"], str)
    assert sample["target_site_index"] == 0
    assert isinstance(sample["target_site_index"], int)
    time_per_spectrum = sample.get("time_per_spectrum")
    assert time_per_spectrum is not None
    assert time_per_spectrum == pytest.approx(0.4, rel=1e-5)


def test_iteration_yields_every_sample_in_order_and_can_restart(written):
    reader_cls, path, _ = written
    with reader_cls(path) as reader:
        first = [s["sample_id"] for s in reader]
        second = [s["sample_id"] for s in reader]
    assert first == second == [f"sample_{i}" for i in range(7)]


def test_get_all_stacks_rows_in_order(written):
    reader_cls, path, batch = written
    with reader_cls(path) as reader:
        everything = reader.get_all()

    assert everything["prediction"].shape == (7, N_ENERGIES)
    np.testing.assert_allclose(everything["prediction"], batch["prediction"], rtol=1e-6)
    assert list(everything["sample_id"]) == [f"sample_{i}" for i in range(7)]
    np.testing.assert_array_equal(everything["target_site_index"], batch["target_site_index"])


def test_get_all_does_not_exhaust_iteration(written):
    reader_cls, path, _ = written
    with reader_cls(path) as reader:
        reader.get_all()
        assert len(list(reader)) == 7


@pytest.mark.parametrize("index", [-1, 7, 100])
def test_out_of_range_index_raises(written, index):
    reader_cls, path, _ = written
    with reader_cls(path) as reader:
        with pytest.raises(IndexError, match="out of range"):
            reader[index]


def test_readers_do_not_provide_structures_by_default(written):
    reader_cls, path, _ = written
    with reader_cls(path) as reader:
        assert reader.provides_structures() is False


@pytest.mark.parametrize("writer_cls, reader_cls", WRITER_READER_PAIRS, ids=PAIR_IDS)
def test_missing_target_site_index_is_reported_as_none(writer_cls, reader_cls, tmp_path):
    batch = make_batch(2)
    del batch["target_site_index"]
    writer = writer_cls(tmp_path, buffer_size=10)
    writer.add(batch)
    writer.close()

    with reader_cls(tmp_path) as reader:
        assert reader[0]["target_site_index"] is None


@pytest.mark.parametrize("writer_cls, reader_cls", WRITER_READER_PAIRS, ids=PAIR_IDS)
def test_optional_uncertainty_field_round_trips(writer_cls, reader_cls, tmp_path):
    std = np.full((2, N_ENERGIES), 0.25, dtype=np.float32)
    writer = writer_cls(tmp_path, buffer_size=10)
    writer.add(make_batch(2, prediction_std=std))
    writer.close()

    with reader_cls(tmp_path) as reader:
        np.testing.assert_allclose(reader[1]["prediction_std"], std[1])


def test_hdf5_reader_close_is_idempotent(tmp_path):
    writer = HDF5Writer(tmp_path, buffer_size=10)
    writer.add(make_batch(1))
    writer.close()

    reader = HDF5Reader(tmp_path)
    reader.close()
    reader.close()
    with pytest.raises(RuntimeError, match="not properly initialized"):
        reader[0]


###############################################################################
############################### READER VALIDATION #############################
###############################################################################


def test_hdf5_reader_requires_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
        HDF5Reader(tmp_path)


def test_hdf5_reader_requires_the_predictions_group(tmp_path):
    with h5py.File(tmp_path / "predictions.h5", "w") as h5:
        h5.create_group("other")
    with pytest.raises(ValueError, match="No 'predictions' group"):
        HDF5Reader(tmp_path)


@pytest.mark.parametrize("reader_cls", [NumpyReader, JSONReader])
class TestFileBasedReaderValidation:
    def test_missing_directory(self, reader_cls, tmp_path):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            reader_cls(tmp_path / "missing")

    def test_path_must_be_a_directory(self, reader_cls, tmp_path):
        file = tmp_path / "file.txt"
        file.write_text("x")
        with pytest.raises(ValueError, match="not a directory"):
            reader_cls(file)

    def test_directory_without_samples(self, reader_cls, tmp_path):
        with pytest.raises(FileNotFoundError, match="No sample_"):
            reader_cls(tmp_path)


###############################################################################
############################## FORMAT DETECTION ################################
###############################################################################


@pytest.mark.parametrize("writer_cls, reader_cls", WRITER_READER_PAIRS, ids=PAIR_IDS)
def test_format_is_detected_from_directory_contents(writer_cls, reader_cls, tmp_path):
    writer = writer_cls(tmp_path, buffer_size=10)
    writer.add(make_batch(1))
    writer.close()
    assert detect_prediction_format(tmp_path) is reader_cls


def test_detecting_a_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Predictions path not found"):
        detect_prediction_format(tmp_path / "missing")


def test_detecting_an_unrecognised_directory_raises(tmp_path):
    (tmp_path / "unrelated.txt").write_text("x")
    with pytest.raises(ValueError, match="Could not detect prediction format"):
        detect_prediction_format(tmp_path)


###############################################################################
############################## STRUCTURE MATCHING #############################
###############################################################################


@pytest.fixture
def hdf5_predictions(tmp_path):
    writer = HDF5Writer(tmp_path, buffer_size=10)
    writer.add(make_batch(4))
    writer.close()
    return tmp_path


@pytest.fixture
def datasource() -> ListDataSource:
    # Deliberately not in prediction order: matching must be by sample_id, not position.
    ids = ["sample_3", "sample_0", "sample_2", "sample_1"]
    return ListDataSource([make_molecule(i) for i in ids])


@pytest.mark.parametrize("preload_structures", [False, True])
def test_structures_are_matched_by_sample_id(hdf5_predictions, datasource, preload_structures):
    with StructureMatchedPredictionReader(
        HDF5Reader(hdf5_predictions), datasource, preload_structures=preload_structures
    ) as reader:
        assert reader.provides_structures() is True
        assert len(reader) == 4
        for index in range(4):
            sample = reader[index]
            structure = sample.get("structure")
            assert structure is not None
            assert structure.properties["sample_id"] == sample["sample_id"] == f"sample_{index}"


def test_structure_matching_leaves_bulk_loading_to_the_wrapped_reader(hdf5_predictions, datasource):
    with StructureMatchedPredictionReader(HDF5Reader(hdf5_predictions), datasource) as reader:
        assert "structure" not in reader.get_all()


def test_unknown_sample_id_is_reported_with_its_index(hdf5_predictions):
    partial = ListDataSource([make_molecule("sample_0")])
    with StructureMatchedPredictionReader(HDF5Reader(hdf5_predictions), partial) as reader:
        assert reader[0]["sample_id"] == "sample_0"
        with pytest.raises(ValueError, match="index 1 references unknown sample_id 'sample_1'"):
            reader[1]


def test_datasource_entries_must_define_a_sample_id(hdf5_predictions):
    with pytest.raises(ValueError, match="must define properties\\['sample_id'\\]"):
        StructureMatchedPredictionReader(HDF5Reader(hdf5_predictions), ListDataSource([make_molecule(None)]))


def test_duplicate_sample_ids_in_the_datasource_are_rejected(hdf5_predictions):
    duplicated = ListDataSource([make_molecule("sample_0"), make_molecule("sample_0")])
    with pytest.raises(ValueError, match="duplicate sample_id 'sample_0'"):
        StructureMatchedPredictionReader(HDF5Reader(hdf5_predictions), duplicated)


###############################################################################
################################ build_prediction_reader ######################
###############################################################################


def test_factory_returns_the_detected_plain_reader(hdf5_predictions):
    with build_prediction_reader(hdf5_predictions) as reader:
        assert type(reader) is HDF5Reader


def test_factory_wraps_the_reader_when_a_datasource_is_given(hdf5_predictions, datasource):
    with build_prediction_reader(hdf5_predictions, datasource=datasource) as reader:
        assert isinstance(reader, StructureMatchedPredictionReader)
        assert reader.provides_structures()


def test_factory_preload_serves_samples_from_memory(hdf5_predictions):
    with build_prediction_reader(hdf5_predictions, preload=True) as reader:
        assert isinstance(reader, PreloadedPredictionReader)
        assert len(reader) == 4
        assert reader[2]["sample_id"] == "sample_2"
        assert reader[2] is reader[2], "preloaded samples are served from memory"
        with pytest.raises(IndexError):
            reader[4]


def test_factory_preload_with_a_datasource_keeps_structures(hdf5_predictions, datasource):
    with build_prediction_reader(hdf5_predictions, datasource=datasource, preload=True) as reader:
        assert reader.provides_structures()
        structure = reader[1].get("structure")
        assert structure is not None
        assert structure.properties["sample_id"] == "sample_1"


def test_factory_propagates_detection_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_prediction_reader(tmp_path / "missing")
