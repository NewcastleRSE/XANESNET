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

from xanesnet.utils import ConfigError

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train")
    sub.add_parser("infer")

    args_namespace, remaining = parser.parse_known_args(args)

    if args_namespace.command == "train":
        # Dispatching to training mode

        from .train import main

        main(remaining)

    elif args_namespace.command == "infer":
        # Dispatching to inference mode

        from .infer import main

        main(remaining)
    else:
        raise ConfigError(f"Incorrect mode: {args_namespace.command}.")


if __name__ == "__main__":
    main(sys.argv[1:])
