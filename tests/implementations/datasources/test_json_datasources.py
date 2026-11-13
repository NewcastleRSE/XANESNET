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

"""Unit tests for the ``pmgjson`` and ``multipmgjson`` data sources (pymatgen ``Structure``/``Molecule`` JSON files)."""

import json
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Molecule, Structure

from xanesnet.datasources import DataSourceRegistry
from xanesnet.datasources.pmgjson import PMGJSONSource
from xanesnet.utils.exceptions import ResourceError

SPECTRUM = {"energies": [1.0, 2.0], "intensities": [0.5, 0.25]}


def structure(spectrum_key: str = "spectrum") -> Structure:
    result = Structure(Lattice.cubic(4.0), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    result.add_site_property(spectrum_key, [SPECTRUM, None])
    return result


def molecule(spectrum_key: str = "spectrum") -> Molecule:
    result = Molecule(["O", "H"], [[0, 0, 0], [0, 0, 1]])
    result.add_site_property(spectrum_key, [SPECTRUM, None])
    return result


def dump(path: Path, entry: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry))
    return path


def create(root: Path, spectrum_key: str = "spectrum"):
    return DataSourceRegistry.create(
        "pmgjson",
        datasource_type="pmgjson",
        json_path=str(root),
        spectrum_key=spectrum_key,
    )


###############################################################################
#################################### PMGJSON ##################################
###############################################################################


def test_structures_and_molecules_are_loaded_by_their_class_tag(tmp_path):
    dump(tmp_path / "b.json", molecule().as_dict())
    dump(tmp_path / "a.json", structure().as_dict())

    source = create(tmp_path)

    assert source.sample_ids == ["a", "b"]
    assert isinstance(source[0], Structure)
    assert isinstance(source[1], Molecule)
    assert [item.properties["sample_id"] for item in source] == ["a", "b"]


def test_non_json_files_are_ignored(tmp_path):
    dump(tmp_path / "a.json", structure().as_dict())
    (tmp_path / "notes.txt").write_text("not a sample")

    assert len(create(tmp_path)) == 1


def test_a_custom_spectrum_key_is_renamed_to_spectrum(tmp_path):
    dump(tmp_path / "a.json", structure("xanes").as_dict())

    sample = create(tmp_path, spectrum_key="xanes")[0]

    assert "xanes" not in sample.site_properties
    assert sample.site_properties["spectrum"][0] == SPECTRUM


def test_a_spectrum_key_the_file_does_not_have_leaves_the_sample_untouched(tmp_path):
    dump(tmp_path / "a.json", structure().as_dict())

    sample = create(tmp_path, spectrum_key="absent")[0]

    assert sample.site_properties["spectrum"][0] == SPECTRUM


@pytest.mark.parametrize("class_tag", [None, "Unsupported"])
def test_entries_with_a_missing_or_unknown_class_tag_fall_back_to_structure_parsing(tmp_path, caplog, class_tag):
    entry = structure().as_dict()
    entry.pop("@class")
    if class_tag is not None:
        entry["@class"] = class_tag
    dump(tmp_path / "a.json", entry)

    with caplog.at_level("WARNING"):
        sample = create(tmp_path)[0]

    assert isinstance(sample, Structure)
    assert "Attempting fallback parsing" in caplog.text


def test_untagged_molecules_fall_back_to_molecule_parsing(tmp_path):
    entry = molecule().as_dict()
    entry.pop("@class")
    dump(tmp_path / "a.json", entry)

    assert isinstance(create(tmp_path)[0], Molecule)


def test_untagged_entries_that_are_neither_structure_nor_molecule_are_reported(tmp_path):
    dump(tmp_path / "a.json", {"unexpected": True})

    with pytest.raises(ResourceError, match="Unsupported JSON object type"):
        create(tmp_path)[0]


def test_json_documents_must_be_objects(tmp_path):
    dump(tmp_path / "a.json", [1, 2, 3])

    with pytest.raises(ResourceError, match="Expected a JSON object"):
        PMGJSONSource.load_json(tmp_path / "a.json")


@pytest.mark.parametrize("layout", ["missing", "empty"])
def test_unusable_directories_are_rejected(tmp_path, layout):
    root = tmp_path / "root"
    if layout == "empty":
        root.mkdir()

    with pytest.raises(ResourceError):
        create(root)


###############################################################################
################################# MULTIPMGJSON ################################
###############################################################################


@pytest.fixture
def multi(tmp_path):
    dump(tmp_path / "set_b" / "y.json", structure().as_dict())
    dump(tmp_path / "set_b" / "x.json", structure().as_dict())
    dump(tmp_path / "set_a" / "z.json", molecule().as_dict())
    return DataSourceRegistry.create(
        "multipmgjson", datasource_type="multipmgjson", root_path=str(tmp_path), spectrum_key="spectrum"
    )


def test_json_samples_are_grouped_by_subdirectory_with_a_stable_subdirectory_id(multi):
    samples = list(multi)

    assert len(multi) == 3
    assert [(s.properties["subdir_name"], s.properties["sample_id"]) for s in samples] == [
        ("set_a", "z"),
        ("set_b", "x"),
        ("set_b", "y"),
    ]
    assert [s.properties["subdir_id"] for s in samples] == [0, 1, 1]


def test_multi_json_renames_the_configured_spectrum_key(tmp_path):
    dump(tmp_path / "set" / "a.json", structure("xanes").as_dict())

    source = DataSourceRegistry.create(
        "multipmgjson", datasource_type="multipmgjson", root_path=str(tmp_path), spectrum_key="xanes"
    )

    assert source[0].site_properties["spectrum"][0] == SPECTRUM
    assert "xanes" not in source[0].site_properties


def test_empty_subdirectories_are_skipped_with_a_warning(tmp_path, caplog):
    dump(tmp_path / "good" / "a.json", structure().as_dict())
    (tmp_path / "empty").mkdir()

    with caplog.at_level("WARNING"):
        source = DataSourceRegistry.create(
            "multipmgjson", datasource_type="multipmgjson", root_path=str(tmp_path), spectrum_key="spectrum"
        )

    assert list(source.sample_ids) == ["good"]
    assert "empty" in caplog.text


@pytest.mark.parametrize("layout", ["missing_root", "no_subdirectories", "no_json_anywhere"])
def test_unusable_roots_are_rejected(tmp_path, layout):
    root = tmp_path / "root"
    if layout != "missing_root":
        root.mkdir()
    if layout == "no_json_anywhere":
        (root / "s").mkdir()

    with pytest.raises(ResourceError):
        DataSourceRegistry.create(
            "multipmgjson", datasource_type="multipmgjson", root_path=str(root), spectrum_key="spectrum"
        )
