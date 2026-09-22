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

"""Split a directory of pymatgen JSON files into trainval and test subsets."""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Molecule, Structure

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xanesnet.datasources.pmgjson import PMGJSONSource
from xanesnet.utils.exceptions import ResourceError
from xanesnet.utils.filesystem import copy_file
from xanesnet.utils.logger import setup_logging

TRAINVAL_DIRNAME: str = "trainval"
TEST_DIRNAME: str = "test"
MANIFEST_FILENAME: str = "split_manifest.json"


def absorber_elements(pmg_obj: Structure | Molecule) -> tuple[str, ...]:
    """Return the sorted unique absorber elements of one pymatgen object.

    Absorber (target) sites are the sites carrying a spectrum entry in the
    ``"spectrum"`` site property, which ``PMGJSONSource`` populates from the
    configured ``spectrum_key``. Sites without a spectrum (``None``) are not
    absorber sites.

    Args:
        pmg_obj: Pymatgen ``Structure`` or ``Molecule`` loaded from a JSON
            file.

    Returns:
        Sorted tuple of unique element symbols of the absorber sites. Empty
        when the object carries no spectrum entries.
    """
    if "spectrum" not in pmg_obj.site_properties:
        return ()

    elements: set[str] = set()
    for site, spectrum in zip(pmg_obj, pmg_obj.site_properties["spectrum"]):
        if spectrum is None:
            continue
        elements.add(site.specie.symbol)
    return tuple(sorted(elements))


def _collect_labels(datasource: PMGJSONSource) -> tuple[list[str], list[tuple[str, ...]]]:
    """Collect sample ids and absorber-element labels from a datasource.

    Samples without any spectrum entry are logged and excluded from the
    split.

    Args:
        datasource: PMG-JSON datasource providing normalized ``"spectrum"``
            site properties.

    Returns:
        Tuple ``(sample_ids, labels)`` where ``labels[i]`` is the sorted
        tuple of absorber element symbols for ``sample_ids[i]``.
    """
    sample_ids: list[str] = []
    labels: list[tuple[str, ...]] = []
    for pmg_obj in datasource:
        sample_id = str(pmg_obj.properties["sample_id"])
        absorbers = absorber_elements(pmg_obj)
        if not absorbers:
            logging.warning("Sample %s has no spectrum entries; excluded from the split.", sample_id)
            continue
        sample_ids.append(sample_id)
        labels.append(absorbers)
    return sample_ids, labels


def _test_size(num_samples: int, test_fraction: float) -> int:
    """Return the number of samples reserved for the test split.

    Args:
        num_samples: Total number of samples.
        test_fraction: Target test fraction in ``(0, 1)``.

    Returns:
        ``num_samples * test_fraction`` rounded to the nearest integer and
        clamped to ``[0, num_samples]``.
    """
    return min(max(round(num_samples * test_fraction), 0), num_samples)


def random_indices(num_samples: int, test_size: int) -> tuple[list[int], list[int]]:
    """Partition sample indices into random trainval and test subsets.

    Samples are permuted with ``numpy.random.permutation`` and therefore
    follow the NumPy global seed.

    Args:
        num_samples: Total number of samples.
        test_size: Number of samples assigned to the test split.

    Returns:
        Tuple ``(trainval_indices, test_indices)`` of disjoint zero-based
        indices covering all ``num_samples`` samples.
    """
    permutation = np.random.permutation(num_samples).tolist()
    return permutation[test_size:], permutation[:test_size]


def stratified_indices(
    labels: list[tuple[str, ...]],
    test_fraction: float,
    test_size: int,
) -> tuple[list[int], list[int]]:
    """Partition sample indices with absorber elements balanced across splits.

    Samples are grouped by their absorber-element label. The global test
    count is distributed over the groups with the largest-remainder method,
    so each group is split as close to ``test_fraction`` as possible while
    the global test size is met exactly. Group members are drawn with
    ``numpy.random.choice`` and therefore follow the NumPy global seed.

    Args:
        labels: Absorber-element label for each sample, aligned with sample
            order.
        test_fraction: Target test fraction in ``(0, 1)``.
        test_size: Number of samples assigned to the test split overall.

    Returns:
        Tuple ``(trainval_indices, test_indices)`` of disjoint zero-based
        sample indices.
    """
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(idx)

    # Allocate the global test count over groups (largest-remainder method).
    allocations: dict[tuple[str, ...], int] = {}
    remainders: list[tuple[float, tuple[str, ...]]] = []
    for label, members in groups.items():
        exact = test_fraction * len(members)
        base = int(exact)
        allocations[label] = base
        remainders.append((exact - base, label))

    allocated = sum(allocations.values())
    for _, label in sorted(remainders, key=lambda item: item[0], reverse=True):
        if allocated >= test_size:
            break
        allocations[label] += 1
        allocated += 1

    # Draw test members within each group.
    test_indices: list[int] = []
    for label, members in groups.items():
        count = allocations[label]
        if count > 0:
            test_indices.extend(np.random.choice(members, size=count, replace=False).tolist())

    test_set = set(test_indices)
    trainval_indices = [idx for idx in range(len(labels)) if idx not in test_set]
    return trainval_indices, test_indices


def _count_elements(labels: list[tuple[str, ...]]) -> dict[str, int]:
    """Count samples per absorber element.

    A sample with several absorber elements contributes to each of them.

    Args:
        labels: Absorber-element labels.

    Returns:
        Mapping from element symbol to the number of samples containing it.
    """
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        for element in label:
            counts[element] += 1
    return dict(counts)


def _copy_split(dest_dir: Path, source_dir: Path, sample_ids: list[str], indices: list[int]) -> None:
    """Copy the JSON files of one split into ``dest_dir``.

    Args:
        dest_dir: Destination directory, created if missing.
        source_dir: Directory containing the source JSON files.
        sample_ids: Sample identifier (file stem) for each sample.
        indices: Zero-based indices of the samples to copy.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        copy_file(source_dir / f"{sample_ids[idx]}.json", dest_dir)


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Write the split manifest as JSON.

    Args:
        manifest_path: Destination path for the manifest file.
        manifest: Manifest dictionary to serialize.
    """
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments without the executable name.

    Returns:
        Parsed namespace for data splitting.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", type=str, required=True, help="Directory containing the PMG-JSON files.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory; trainval/ and test/ subdirectories are created inside it.",
    )
    parser.add_argument(
        "-f",
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for the test split (default: 0.2).",
    )
    parser.add_argument(
        "-k",
        "--spectrum-key",
        type=str,
        default="XANES",
        help="Site-property key under which spectra are stored (default: XANES).",
    )
    parser.add_argument(
        "--stratify-absorbers",
        action="store_true",
        help="Balance absorber elements between the trainval and test split (default: off).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the NumPy random generator (default: 42).")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    """Run the PMG-JSON data-splitting command-line interface.

    Args:
        argv: Command-line arguments without the executable name.
    """
    setup_logging(logging.INFO)
    args = parse_args(argv)

    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError(f"Test fraction must lie in (0, 1), got {args.test_fraction}")
    if args.seed < 0:
        raise ValueError(f"Seed must be non-negative, got {args.seed}")

    np.random.seed(args.seed)
    logging.info("Random seed: %d", args.seed)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    datasource = PMGJSONSource(
        datasource_type="pmgjson",
        json_path=str(input_dir),
        spectrum_key=args.spectrum_key,
    )
    logging.info("Found %d JSON samples in %s", len(datasource), input_dir)

    sample_ids, labels = _collect_labels(datasource)
    if not sample_ids:
        raise ResourceError(f"No spectra found under site-property key {args.spectrum_key!r} in {input_dir}")

    excluded_ids = sorted(set(datasource.sample_ids) - set(sample_ids))
    if excluded_ids:
        logging.warning(
            "Excluded %d sample(s) without spectrum entries: %s",
            len(excluded_ids),
            ", ".join(excluded_ids),
        )

    test_size = _test_size(len(sample_ids), args.test_fraction)
    if args.stratify_absorbers:
        trainval_idx, test_idx = stratified_indices(labels, args.test_fraction, test_size)
    else:
        trainval_idx, test_idx = random_indices(len(sample_ids), test_size)

    if not trainval_idx or not test_idx:
        logging.warning("One split is empty; consider a different test fraction.")

    trainval_ids = [sample_ids[i] for i in trainval_idx]
    test_ids = [sample_ids[i] for i in test_idx]

    total_counts = _count_elements(labels)
    trainval_counts = _count_elements([labels[i] for i in trainval_idx])
    test_counts = _count_elements([labels[i] for i in test_idx])

    logging.info("Absorber element distribution across splits:")
    for element in sorted(total_counts):
        logging.info(
            "  %-3s: total=%4d  trainval=%4d  test=%4d  (test fraction %.3f)",
            element,
            total_counts[element],
            trainval_counts.get(element, 0),
            test_counts.get(element, 0),
            test_counts.get(element, 0) / total_counts[element],
        )

    trainval_dir = output_dir / TRAINVAL_DIRNAME
    test_dir = output_dir / TEST_DIRNAME
    for split_dir in (trainval_dir, test_dir):
        if split_dir.exists() and any(split_dir.iterdir()):
            logging.warning("Destination directory %s is not empty; existing files may be overwritten.", split_dir)

    _copy_split(trainval_dir, input_dir, sample_ids, trainval_idx)
    _copy_split(test_dir, input_dir, sample_ids, test_idx)
    logging.info("Wrote %d trainval and %d test JSON files under %s", len(trainval_ids), len(test_ids), output_dir)

    manifest = {
        "input_dir": str(input_dir),
        "spectrum_key": args.spectrum_key,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "stratify_absorbers": args.stratify_absorbers,
        "excluded": excluded_ids,
        "trainval": sorted(f"{stem}.json" for stem in trainval_ids),
        "test": sorted(f"{stem}.json" for stem in test_ids),
        "absorber_counts": {
            "total": total_counts,
            "trainval": trainval_counts,
            "test": test_counts,
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    _write_manifest(manifest_path, manifest)
    logging.info("Wrote split manifest to %s", manifest_path)


if __name__ == "__main__":
    main(sys.argv[1:])
