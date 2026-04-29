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

"""Debug entry points for development purposes."""

from xanesnet.cli import main


def run_debug_train() -> None:
    debug_args = [
        "train",
        "-i",
        "./configs/<...>.yaml",  # Insert path to training config
        "-n",
        "<...>",  # Insert name for training run
        # "--tensorboard",  # Enable TensorBoard logging
        "--yes",  # Skip all prompts with yes (use with caution!)
    ]

    print("Running in debug mode with the following arguments:")
    print(debug_args)

    main(debug_args)


def run_debug_infer() -> None:
    debug_args = [
        "infer",
        "-i",
        "./configs/<...>.yaml",  # Insert path to inference config
        "-m",
        "./runs/<...>/models/final.pth",  # Insert path to trained model (final.pth)
        "-n",
        "<...>",  # Insert name for inference run
        "--yes",  # Skip all prompts with yes (use with caution!)
    ]

    print("Running in debug mode with the following arguments:")
    print(debug_args)

    main(debug_args)


def run_debug_analyze() -> None:
    debug_args = [
        "analyze",
        "-i",
        "./configs/<...>.yaml",  # Insert path to analysis config
        "-p",
        "./runs/<...>/predictions",  # Insert path to predictions directory
        "-p",
        "./runs/<...>/predictions",
        "-n",
        "<...>",  # Insert name for analysis run
        "--yes",  # Skip all prompts with yes (use with caution!)
    ]

    print("Running in debug mode with the following arguments:")
    print(debug_args)

    main(debug_args)


if __name__ == "__main__":
    # Uncomment the desired debug function to run it.

    run_debug_train()
    # run_debug_infer()
    # run_debug_analyze()
