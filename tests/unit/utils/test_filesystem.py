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

"""Unit tests for filesystem helpers: listing, run-directory creation and file copying."""

import re
from pathlib import Path

import pytest

from xanesnet.utils.filesystem import (
    copy_file,
    create_run_dir,
    create_subfolders,
    list_files,
    list_filestems,
    list_subdir_stems,
    list_subdirs,
)


@pytest.fixture
def populated_dir(tmp_path: Path) -> Path:
    """Directory with visible files, a hidden file and sub-directories."""
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "c.txt").write_text("c")
    (tmp_path / ".hidden.json").write_text("{}")
    (tmp_path / "zeta").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / ".cache").mkdir()
    return tmp_path


###############################################################################
################################### LISTING ###################################
###############################################################################


def test_list_files_returns_visible_files_only(populated_dir):
    assert {p.name for p in list_files(populated_dir)} == {"a.json", "b.json", "c.txt"}


@pytest.mark.parametrize("suffixes", [".json", (".json",)], ids=["str", "tuple"])
def test_list_files_filters_by_suffix(populated_dir, suffixes):
    assert {p.name for p in list_files(populated_dir, suffixes=suffixes)} == {"a.json", "b.json"}


def test_list_files_can_strip_the_extension(populated_dir):
    assert {p.name for p in list_files(populated_dir, with_ext=False)} == {"a", "b", "c"}


def test_list_files_ignores_directories(populated_dir):
    assert all(p.is_file() for p in list_files(populated_dir))


def test_list_filestems(populated_dir):
    assert sorted(list_filestems(populated_dir, suffixes=".json")) == ["a", "b"]


def test_list_subdirs_skips_hidden_directories(populated_dir):
    assert {p.name for p in list_subdirs(populated_dir)} == {"alpha", "zeta"}


def test_list_subdir_stems_are_sorted(populated_dir):
    assert list_subdir_stems(populated_dir) == ["alpha", "zeta"]


def test_listing_an_empty_directory_gives_empty_results(tmp_path):
    assert list_files(tmp_path) == []
    assert list_subdirs(tmp_path) == []


###############################################################################
################################ create_run_dir ###############################
###############################################################################


def test_count_mode_starts_at_zero_and_increments(tmp_path):
    first = create_run_dir(tmp_path, name="train")
    second = create_run_dir(tmp_path, name="train")
    assert (first.name, second.name) == ("train_000", "train_001")
    assert first.is_dir() and second.is_dir()


def test_count_mode_continues_after_the_highest_existing_index(tmp_path):
    (tmp_path / "train_004").mkdir()
    (tmp_path / "train_not_a_number").mkdir()
    assert create_run_dir(tmp_path, name="train").name == "train_005"


def test_count_mode_counts_each_name_independently(tmp_path):
    (tmp_path / "train_007").mkdir()
    assert create_run_dir(tmp_path, name="infer").name == "infer_000"


def test_count_mode_defaults_to_run_name(tmp_path):
    assert create_run_dir(tmp_path).name == "run_000"


def test_count_mode_creates_missing_parent_directories(tmp_path):
    run_dir = create_run_dir(tmp_path / "nested" / "runs", name="x")
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path / "nested" / "runs"


def test_base_dir_may_be_given_as_a_string(tmp_path):
    assert create_run_dir(str(tmp_path), name="x").parent == tmp_path


@pytest.mark.parametrize("name, pattern", [(None, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"), ("exp", r"_exp$")])
def test_time_mode_uses_a_timestamp_with_optional_suffix(tmp_path, name, pattern):
    run_dir = create_run_dir(tmp_path, name=name, mode="time")
    assert re.search(pattern, run_dir.name)
    assert run_dir.is_dir()


def test_invalid_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="Invalid mode"):
        create_run_dir(tmp_path, mode="bogus")


###############################################################################
############################### create_subfolders #############################
###############################################################################


def test_create_subfolders_creates_and_maps_every_name(tmp_path):
    paths = create_subfolders(tmp_path, ["models", "checkpoints"])
    assert paths == {"models": tmp_path / "models", "checkpoints": tmp_path / "checkpoints"}
    assert all(p.is_dir() for p in paths.values())


def test_create_subfolders_is_idempotent(tmp_path):
    create_subfolders(tmp_path, ["models"])
    assert create_subfolders(tmp_path, ["models"]) == {"models": tmp_path / "models"}


def test_create_subfolders_requires_an_existing_parent(tmp_path):
    with pytest.raises(FileNotFoundError, match="Parent directory does not exist"):
        create_subfolders(tmp_path / "missing", ["models"])


###############################################################################
#################################### copy_file ################################
###############################################################################


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    src = tmp_path / "src" / "config.yaml"
    src.parent.mkdir()
    src.write_text("seed: 1\n")
    return src


@pytest.fixture
def dest_dir(tmp_path: Path) -> Path:
    dst = tmp_path / "dst"
    dst.mkdir()
    return dst


def test_copy_preserves_name_and_content(source_file, dest_dir):
    copied = copy_file(source_file, dest_dir)
    assert copied == dest_dir / "config.yaml"
    assert copied.read_text() == "seed: 1\n"
    assert source_file.exists()


def test_copy_can_rename(source_file, dest_dir):
    assert copy_file(source_file, dest_dir, new_name="renamed.yaml") == dest_dir / "renamed.yaml"


def test_copy_accepts_string_paths(source_file, dest_dir):
    assert copy_file(str(source_file), str(dest_dir)).exists()


def test_copy_rejects_a_disallowed_suffix(source_file, dest_dir):
    with pytest.raises(ValueError, match="suffix not allowed"):
        copy_file(source_file, dest_dir, allowed_suffixes={".json"})
    assert list(dest_dir.iterdir()) == []


def test_copy_accepts_an_allowed_suffix(source_file, dest_dir):
    assert copy_file(source_file, dest_dir, allowed_suffixes={".yaml", ".yml"}).exists()


@pytest.mark.parametrize("which", ["missing", "directory"])
def test_copy_requires_a_regular_source_file(tmp_path, dest_dir, which):
    src = tmp_path / "missing.yaml" if which == "missing" else tmp_path
    with pytest.raises(FileNotFoundError, match="Source file does not exist"):
        copy_file(src, dest_dir)


def test_copy_requires_an_existing_destination_directory(source_file, tmp_path):
    with pytest.raises(FileNotFoundError, match="Destination directory does not exist"):
        copy_file(source_file, tmp_path / "nowhere")
