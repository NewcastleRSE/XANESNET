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

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################


def main(args: list[str]):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train")
    sub.add_parser("infer")

    args, remaining = parser.parse_known_args(args)

    if args.command == "train":
        # Dispatching to training mode

        from .train import main

        main(remaining)

    elif args.command == "infer":
        # Dispatching to inference mode

        from .infer import main

        main(remaining)
    else:
        raise ValueError(f"Incorrect mode: {args.command}.")


if __name__ == "__main__":
    main(sys.argv[1:])
