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

"""Inspect saved prediction files with the available XANESNET prediction readers."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from xanesnet.serialization.prediction_readers import (
    HDF5Reader,
    JSONReader,
    NumpyReader,
    PredictionReader,
)


def format_value(value: Any, full: bool = False) -> str:
    """Format a prediction value for compact terminal output.

    Args:
        value: Value returned by a prediction reader.
        full: Whether ndarray values should be printed in full.

    Returns:
        Human-readable value summary.
    """

    if isinstance(value, np.ndarray):
        s = f"shape={value.shape}, dtype={value.dtype}"
        if full:
            s += f"\n{value}"
        elif value.size < 10:
            s += f", val={value}"
        elif np.issubdtype(value.dtype, np.number):
            flat = value.flatten()
            s += f", min={flat.min():.4f}, max={flat.max():.4f}, mean={flat.mean():.4f}"
            s += f", head={flat[:3]}"
        else:
            flat = value.flatten()
            s += f", head={flat[:3]}"
        return s
    return f"{type(value)} = {value}"


def test_reader(reader_cls: type[PredictionReader], path: Path, show_sample: bool = False) -> None:
    """Exercise one prediction reader implementation against a directory.

    Args:
        reader_cls: Prediction reader class to instantiate.
        path: Directory containing prediction files for that reader.
        show_sample: Whether to print every field from the first sample.
    """

    print(f"\n{'='*60}")
    print(f"Testing {reader_cls.__name__} at: {path}")
    print(f"{'='*60}")

    try:
        with reader_cls(path) as reader:
            total_samples = len(reader)
            print(f"Total samples (len): {total_samples}")

            if total_samples == 0:
                print("Warning: No samples found.")
                return

            if show_sample:
                print("\n--- FULL SAMPLE VIEW (Index 0) ---")
                sample0 = reader[0]
                for key, value in sample0.items():
                    print(f"\n[{key}]")
                    print(format_value(value, full=True))
                print("\n" + "-" * 40)

            print("\n--- Testing Iteration (first 3) ---")
            start_time = time.perf_counter()
            for i, sample in enumerate(reader):
                if i >= 3:
                    break
                print(f"Sample {i}:")
                for key, value in sample.items():
                    print(f"  {key}: {format_value(value)}")
            end_time = time.perf_counter()
            print(f"\nIteration time (first {min(3, total_samples)}): {(end_time - start_time)*1000:.4f} ms")

            idx = total_samples - 1
            print(f"\n--- Testing Random Access (index {idx}) ---")

            start_time = time.perf_counter()
            last_sample = reader[idx]
            end_time = time.perf_counter()

            print(f"Sample {idx} keys: {list(last_sample.keys())}")
            print(f"Random access time: {(end_time - start_time)*1000:.4f} ms")

            print("\n--- Testing get_all() ---")

            start_time = time.perf_counter()
            all_data = reader.get_all()
            end_time = time.perf_counter()

            for key, value in all_data.items():
                print(f"  {key}: {format_value(value)}")

            print(f"\nget_all() time ({total_samples} samples): {(end_time - start_time)*1000:.4f} ms")

    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback

        traceback.print_exc()


def main() -> None:
    """Run the prediction-reader smoke-test command-line interface."""

    parser = argparse.ArgumentParser(description="Test XANESNET Prediction Readers")
    parser.add_argument("path", type=str, help="Path to the directory containing prediction files")
    parser.add_argument("--show-sample", action="store_true", help="Print the full content of the first sample")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path '{path}' does not exist.")
        return

    print(f"Inspecting directory: {path}")

    readers_to_test: list[type[PredictionReader]] = []

    if (path / "predictions.h5").exists():
        print("-> Found 'predictions.h5', testing HDF5Reader.")
        readers_to_test.append(HDF5Reader)

    if list(path.glob("sample_*.npz")):
        print("-> Found 'sample_*.npz' files, testing NumpyReader.")
        readers_to_test.append(NumpyReader)

    if list(path.glob("sample_*.json")):
        print("-> Found 'sample_*.json' files, testing JSONReader.")
        readers_to_test.append(JSONReader)

    if not readers_to_test:
        print("Error: Could not detect valid prediction files in directory.")
        print("Expected one of:")
        print("  - predictions.h5")
        print("  - sample_*.npz")
        print("  - sample_*.json")
        return

    for reader_cls in readers_to_test:
        test_reader(reader_cls, path, show_sample=args.show_sample)


if __name__ == "__main__":
    main()
