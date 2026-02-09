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

from xanesnet.cli import main

# DEBUG ENTRY POINTS FOR DEVELOPMENT PURPOSES ONLY


def run_debug_train() -> None:
    debug_args = [
        "train",
        "-i",
        "./configs/in_mlp.yaml",
        "-n",
        "debug",
    ]

    print("Running in debug mode with the following arguments:")
    print(debug_args)

    main(debug_args)


def run_debug_infer() -> None:
    debug_args = [
        "infer",
        "-i",
        "./configs/in_mlp_infer.yaml",
        "-m",
        "./runs/2026-02-09_16-22-51_train_debug/models/final.pth",  # Insert path to trained model (final.pth)
        "-n",
        "debug",
    ]

    print("Running in debug mode with the following arguments:")
    print(debug_args)

    main(debug_args)


def run_debug_analyze() -> None:
    debug_args = [
        "analyze",
        "-i",
        "./configs/analyze_example.yaml",
        "-p",
        "./runs/2026-02-04_11-44-39_infer_mlp_single/predictions/",
        "-p",
        "./runs/2026-02-06_08-11-45_infer_mlp_single/predictions",
    ]

    print("Running in debug mode with the following arguments:")
    print(debug_args)

    main(debug_args)


if __name__ == "__main__":
    # run_debug_train()
    run_debug_infer()
    # run_debug_analyze()
