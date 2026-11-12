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

"""Unit tests for the graph datasets and their agreement with the batch processors that consume them."""

import pytest

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.datasets import DatasetRegistry
from xanesnet.serialization.config import Config

from .conftest import ListSource, spectrum, water

RADIUS = Config({"graph_builder_type": "radius", "cutoff": 3.0, "max_num_neighbors": 20})
ATT_RADIUS = Config({"graph_builder_type": "radius", "cutoff": 3.0, "max_num_neighbors": 20})

# dataset type -> (model type, dataset-specific constructor arguments)
GRAPH_DATASETS = {
    "geometrygraph": ("schnet", {"graph_builder": RADIUS, "compute_angles": False}),
    "gemnet": ("gemnet", {"graph_builder": RADIUS, "quadruplets": False}),
    "e3ee": (
        "e3ee",
        {"graph_builder": RADIUS, "att_graph_builder": ATT_RADIUS, "use_path_branch": True, "max_paths_per_structure": 10},
    ),
    "e3ee_full": (
        "e3ee_full",
        {
            "graph_builder": RADIUS,
            "att_graph_builder": ATT_RADIUS,
            "use_path_branch": True,
            "max_paths_per_site": 10,
            "use_target_site_mask": True,
        },
    ),
}


def create(key: str, tmp_path, molecules, **extra):
    kwargs = {
        "dataset_type": key,
        "datasource": ListSource(molecules),
        "root": str(tmp_path / "processed"),
        "preload": False,
        "skip_prepare": False,
        "split_ratios": None,
        "split_indexfile": None,
    }
    kwargs.update(GRAPH_DATASETS.get(key, (None, {}))[1])
    kwargs.update(extra)
    dataset = DatasetRegistry.create(key, **kwargs)
    dataset.prepare()
    return dataset


def molecules():
    return [water("a", [spectrum(), None, None]), water("b", [None, spectrum(2.0), spectrum(3.0)])]


###############################################################################
################################# GEOMETRY GRAPH ##############################
###############################################################################


def test_geometry_graph_samples_are_whole_structures_with_their_target_sites(tmp_path):
    dataset = create("geometrygraph", tmp_path, molecules())

    first, second = dataset[0], dataset[1]

    assert len(dataset) == 2  # one graph per structure, however many target sites it has
    assert first.x.tolist() == [8, 1, 1]
    assert first.target_site_mask.tolist() == [True, False, False]
    assert second.target_site_mask.tolist() == [False, True, True]
    assert second.target_site_index.tolist() == [1, 2]
    assert second.intensities.shape == (2, 3)
    assert first.sample_id == "a"
    assert first.edge_index.shape[0] == 2 and first.edge_index.shape[1] == first.edge_weight.shape[0]


def test_angles_are_only_computed_when_requested(tmp_path):
    plain = create("geometrygraph", tmp_path / "plain", molecules())[0]
    with_angles = create("geometrygraph", tmp_path / "angles", molecules(), compute_angles=True)[0]

    assert "angle" not in plain and "idx_kj" not in plain
    assert with_angles.angle.numel() == with_angles.idx_kj.numel() == with_angles.idx_ji.numel() > 0


def test_structures_without_a_spectrum_are_skipped(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        dataset = create("geometrygraph", tmp_path, [water("none", None), *molecules()])

    assert len(dataset) == 2
    assert "No spectrum found for sample 0 (none)" in caplog.text


def test_a_batch_has_one_spectrum_and_one_sample_id_per_target_site(tmp_path):
    dataset = create("geometrygraph", tmp_path, molecules())

    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert batch.intensities.shape == (3, 3)
    assert batch.energies.shape == (3, 3)
    assert batch.target_site_mask.tolist() == [True, False, False, False, True, True]
    assert batch.sample_id == ["a", "b", "b"]
    assert batch.x.shape == (6,)


def test_batching_shifts_the_edge_indices_of_later_graphs(tmp_path):
    dataset = create("geometrygraph", tmp_path, molecules())
    first, second = dataset[0], dataset[1]

    batch = dataset.collate_fn([first, second])

    assert batch.edge_index[:, : first.edge_index.shape[1]].tolist() == first.edge_index.tolist()
    assert batch.edge_index[:, first.edge_index.shape[1] :].tolist() == (second.edge_index + 3).tolist()
    assert batch.batch.tolist() == [0, 0, 0, 1, 1, 1]


def test_batching_shifts_the_triplet_indices_by_the_number_of_edges(tmp_path):
    dataset = create("geometrygraph", tmp_path, molecules(), compute_angles=True)
    first, second = dataset[0], dataset[1]

    batch = dataset.collate_fn([first, second])

    n_first = first.idx_kj.numel()
    assert batch.idx_kj[n_first:].tolist() == (second.idx_kj + first.edge_index.shape[1]).tolist()
    assert batch.idx_ji[n_first:].tolist() == (second.idx_ji + first.edge_index.shape[1]).tolist()


def test_the_signature_records_the_graph_settings(tmp_path):
    signature = create("geometrygraph", tmp_path, molecules(), compute_angles=True).signature.as_dict()

    assert signature["dataset_type"] == "geometrygraph"
    assert signature["compute_angles"] is True
    assert signature["graph_builder"] == RADIUS.as_dict()


###############################################################################
############################ DATASET / PROCESSOR CONTRACT #####################
###############################################################################


@pytest.mark.parametrize("key", list(GRAPH_DATASETS))
def test_the_batch_processor_of_each_graph_dataset_can_read_its_batches(tmp_path, key):
    model_type = GRAPH_DATASETS[key][0]
    dataset = create(key, tmp_path, molecules())
    processor = BatchProcessorRegistry.create((key, model_type))

    batch = dataset.collate_fn([dataset[i] for i in range(len(dataset))])
    inputs = processor.input_preparation(batch)
    targets = processor.target_preparation(batch)
    elements = processor.element_preparation(batch)

    # quadruplet fields are optional and absent unless the dataset computes them
    required = {name: value for name, value in inputs.items() if not name.startswith(("int_", "id4_", "Kidx4"))}
    assert required
    assert all(value is not None for value in required.values())
    assert targets.dim() == 2 and targets.shape[-1] == 3
    assert elements.shape[0] == targets.shape[0]
    assert len(processor.sample_id_preparation(batch)) == targets.shape[0]
