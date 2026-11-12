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

"""Arrange total-convolution XANES data for the paper workflow."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

VARIANT = "totalconv"
SPLITS = (("trainval", "train"), ("test", "test"))


def _paired_files(group_dir: Path, raw_split: str) -> list[tuple[Path, Path]]:
    xyz_dir = group_dir / f"xyz_{raw_split}"
    spectra_dir = group_dir / f"xanes_{raw_split}"
    if not xyz_dir.is_dir() or not spectra_dir.is_dir():
        raise FileNotFoundError(f"Expected {xyz_dir} and {spectra_dir}.")

    xyz_files = {path.stem: path for path in xyz_dir.iterdir() if path.is_file() and path.suffix == ".xyz"}
    spectra_files = {path.stem: path for path in spectra_dir.iterdir() if path.is_file() and path.suffix == ".txt"}
    missing_spectra = sorted(xyz_files.keys() - spectra_files.keys())
    missing_xyz = sorted(spectra_files.keys() - xyz_files.keys())
    if missing_spectra or missing_xyz:
        details = []
        if missing_spectra:
            details.append(f"missing spectra for {missing_spectra[:5]}")
        if missing_xyz:
            details.append(f"missing XYZ files for {missing_xyz[:5]}")
        raise ValueError(f"Unpaired files in {group_dir} ({raw_split}): {'; '.join(details)}")

    return [(xyz_files[sample_id], spectra_files[sample_id]) for sample_id in sorted(xyz_files)]


def _source_groups(source_root: Path) -> list[tuple[str, dict[str, list[tuple[Path, Path]]]]]:
    groups = []
    element_dirs = sorted(path for path in source_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    for element_dir in element_dirs:
        variant_dir = element_dir / VARIANT
        if not variant_dir.is_dir():
            raise FileNotFoundError(f"Missing {VARIANT} directory for element {element_dir.name}: {variant_dir}")
        pairs_by_split = {output_split: _paired_files(variant_dir, raw_split) for output_split, raw_split in SPLITS}
        groups.append((element_dir.name, pairs_by_split))

    if not groups:
        raise ValueError(f"No element directories found under {source_root}.")
    return groups


def _validate_stale_group(group_dir: Path, source_root: Path) -> None:
    if group_dir.is_symlink() or not group_dir.is_dir():
        raise FileExistsError(f"Refusing to remove non-directory prepared group: {group_dir}")

    for path in group_dir.rglob("*"):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_file() or source_root not in target.parents:
                raise FileExistsError(f"Refusing to remove link outside the source dataset: {path}")
        elif not path.is_dir():
            raise FileExistsError(f"Refusing to remove non-symlink prepared file: {path}")


def _materialize_files(
    source_files: list[Path],
    destination_dir: Path,
    source_root: Path,
    copy_files: bool,
) -> None:
    if destination_dir.is_symlink():
        raise FileExistsError(f"Refusing to write through symlinked directory: {destination_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    expected_names = {path.name for path in source_files}
    unexpected_names = sorted(path.name for path in destination_dir.iterdir() if path.name not in expected_names)
    if unexpected_names:
        raise FileExistsError(
            f"Unexpected files in {destination_dir}: {unexpected_names[:5]}. "
            "Remove stale prepared data before rerunning."
        )

    for source_file in source_files:
        destination_file = destination_dir / source_file.name
        source_target = source_file.resolve()
        if destination_file.is_symlink():
            if destination_file.resolve(strict=True) != source_target:
                raise FileExistsError(f"Existing symlink points elsewhere: {destination_file}")
            if not copy_files:
                continue
            destination_file.unlink()
        elif destination_file.exists():
            if not destination_file.is_file() or not filecmp.cmp(source_file, destination_file, shallow=False):
                raise FileExistsError(f"Refusing to replace existing file: {destination_file}")
            if copy_files:
                continue
            destination_file.unlink()

        if copy_files:
            shutil.copy2(source_file, destination_file)
        else:
            destination_file.symlink_to(source_target)


def prepare_data(source_root: Path, output_root: Path, copy_files: bool = False) -> dict[str, int]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Source and output roots must be separate, non-nested directories.")

    groups = _source_groups(source_root)
    expected_group_names = {
        output_split: {f"{element}_{VARIANT}" for element, pairs_by_split in groups if pairs_by_split[output_split]}
        for output_split, _ in SPLITS
    }
    stale_groups = []
    for output_split, _ in SPLITS:
        split_root = output_root / output_split
        if split_root.is_symlink():
            raise FileExistsError(f"Refusing to write through symlinked directory: {split_root}")
        if not split_root.exists():
            continue
        if not split_root.is_dir():
            raise FileExistsError(f"Expected a prepared split directory: {split_root}")
        for group_dir in split_root.iterdir():
            if group_dir.name in expected_group_names[output_split]:
                if group_dir.is_symlink() or not group_dir.is_dir():
                    raise FileExistsError(f"Expected a prepared group directory: {group_dir}")
                continue
            _validate_stale_group(group_dir, source_root)
            stale_groups.append(group_dir)

    for group_dir in stale_groups:
        shutil.rmtree(group_dir)

    counts = {output_split: 0 for output_split, _ in SPLITS}
    for element, pairs_by_split in groups:
        group_name = f"{element}_{VARIANT}"
        for output_split, pairs in pairs_by_split.items():
            if not pairs:
                continue
            group_root = output_root / output_split / group_name
            _materialize_files([xyz_file for xyz_file, _ in pairs], group_root / "xyz", source_root, copy_files)
            _materialize_files(
                [spectrum_file for _, spectrum_file in pairs], group_root / "spectra", source_root, copy_files
            )
            counts[output_split] += len(pairs)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing element directories with totalconv data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root for the prepared trainval and test data.",
    )
    file_mode = parser.add_mutually_exclusive_group()
    file_mode.add_argument(
        "--copy", dest="copy_files", action="store_true", help="Copy source files into the prepared tree."
    )
    file_mode.add_argument(
        "--symlink", dest="copy_files", action="store_false", help="Symlink source files (the default)."
    )
    parser.set_defaults(copy_files=False)
    args = parser.parse_args()

    try:
        counts = prepare_data(args.source_root, args.output_root, copy_files=args.copy_files)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output_root = args.output_root.resolve()
    file_mode_name = "copies" if args.copy_files else "symlinks"
    print(f"Prepared {counts['trainval']:,} trainval and {counts['test']:,} test pairs in {output_root}.")
    print(f"Included only {VARIANT}; prepared files are {file_mode_name} and source data was not modified.")


if __name__ == "__main__":
    main()
