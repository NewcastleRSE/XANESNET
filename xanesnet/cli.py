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

import argparse
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
      -i, --in_file    Path to input YAML configuration file. (Required)
      --in_model       Path to a pre-trained model directory (optional). ! Not implemented yet. !

  infer    Run inference on data using a trained model.
    Arguments:
      -i, --in_file    Path to input YAML configuration file. (Required)
      -m, --in_model   Path to a trained model .pth file. (Required)

  analyze  Analyze predictions from inference runs.
    Arguments:
      -i, --in_file         Path to input YAML configuration file. (Required)
      -p, --predictions     Path to directory containing predictions. (Required)
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
    print(LOGO)

    if len(args) == 0 or any(arg in ["-h", "--help"] for arg in args):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(add_help=False)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", add_help=False)
    sub.add_parser("infer", add_help=False)
    sub.add_parser("analyze", add_help=False)

    try:
        args_namespace, remaining = parser.parse_known_args(args)
    except argparse.ArgumentError:
        print("Invalid command. Use -h for help.")
        sys.exit(1)

    if args_namespace.command == "train":
        # Dispatching to training mode

        from xanesnet.train import main

        print(TRAIN)

        main(remaining)

    elif args_namespace.command == "infer":
        # Dispatching to inference mode

        from xanesnet.infer import main

        print(INFER)

        main(remaining)

    elif args_namespace.command == "analyze":
        # Dispatching to analyze mode

        from xanesnet.analyze import main

        print(ANALYZE)

        main(remaining)
    else:
        raise ConfigError(f"Incorrect mode: {args_namespace.command}.")


if __name__ == "__main__":
    main(sys.argv[1:])
