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

"""Command-line interface dispatcher for XANESNET (train / infer / analyze)."""

import sys

from xanesnet.utils.exceptions import ConfigError

LOGO = r"""
////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//     ██╗  ██╗ █████╗ ███╗   ██╗███████╗███████╗███╗   ██╗███████╗████████╗      //
//     ╚██╗██╔╝██╔══██╗████╗  ██║██╔════╝██╔════╝████╗  ██║██╔════╝╚══██╔══╝      //
//      ╚███╔╝ ███████║██╔██╗ ██║█████╗  ███████╗██╔██╗ ██║█████╗     ██║         //
//      ██╔██╗ ██╔══██║██║╚██╗██║██╔══╝  ╚════██║██║╚██╗██║██╔══╝     ██║         //
//     ██╔╝ ██╗██║  ██║██║ ╚████║███████╗███████║██║ ╚████║███████╗   ██║         //
//     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝         //
//                                                                                //
////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//                         Deep Learning for Spectroscopy                         //
//                                                                                //
////////////////////////////////////////////////////////////////////////////////////
    """

HELP = """Usage: python xanesnet/cli.py <command> [options]

Commands (mutually exclusive):
  train    Train a model using a configuration file.
    Arguments:
      -i, --in_file       Path to input YAML configuration file. (Required)
      -o, --out_dir       Path to output directory. (Optional, default: ./runs )
      -n, --name          Name for the training run used for logging and saving (Optional).
      -t, --tensorboard   Whether to write training metrics to TensorBoard logs (Optional).
      -y, --yes           Automatically answer yes to confirmation prompts (Optional).

  infer    Run inference on data using a trained model.
    Arguments:
      -i, --in_file       Path to input YAML configuration file. (Required)
      -m, --in_model      Path to a trained model .pth file. (Required)
      -o, --out_dir       Path to output directory. (Optional, default: ./runs )
      -n, --name          Name for the inference run used for logging and saving (Optional).
      -y, --yes           Automatically answer yes to confirmation prompts (Optional).

  analyze  Analyze predictions from inference runs.
    Arguments:
      -i, --in_file       Path to input YAML configuration file. (Required)
      -p, --predictions   Path to directory containing predictions. (Required)
      -o, --out_dir       Path to output directory. (Optional, default: ./runs )
      -n, --name          Name for the analysis run used for logging and saving (Optional).
      -y, --yes           Automatically answer yes to confirmation prompts (Optional).
"""

TRAIN = r"""
 ____ ____ ____ ____ ____ 
||T |||R |||A |||I |||N ||
||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|
"""

INFER = r"""
 ____ ____ ____ ____ ____ 
||I |||N |||F |||E |||R ||
||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|
"""

ANALYZE = r"""
 ____ ____ ____ ____ ____ ____ ____ 
||A |||N |||A |||L |||Y |||Z |||E ||
||__|||__|||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|/__\|/__\|
"""

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################


def main(args: list[str]) -> None:
    """Dispatch to the train, infer, or analyze sub-command.

    Prints the XANESNET logo and routes to the appropriate sub-command entry
    point based on the first positional argument.

    Args:
        args: Raw command-line argument strings (typically ``sys.argv[1:]``).

    Raises:
        ConfigError: If an unrecognised sub-command is supplied.
    """
    print(LOGO)

    if len(args) == 0 or args[0] in ["-h", "--help"]:
        print(HELP)
        sys.exit(0)

    command = args[0]
    remaining = args[1:]

    if command == "train":
        from xanesnet.train import main

        print(TRAIN)

        main(remaining)

    elif command == "infer":
        from xanesnet.infer import main

        print(INFER)

        main(remaining)

    elif command == "analyze":
        from xanesnet.analyze import main

        print(ANALYZE)

        main(remaining)
    else:
        raise ConfigError(f"Incorrect mode: {command}.")


if __name__ == "__main__":
    main(sys.argv[1:])
