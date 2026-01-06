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

import logging
import os
import sys
from argparse import ArgumentParser

import yaml

from xanesnet.core_learn import train
from xanesnet.core_predict import predict
from xanesnet.utils.logger import setup_file_logging, setup_logging

###############################################################################
################################### LOGGING ###################################
###############################################################################

setup_logging(logging.INFO)
setup_file_logging("./")

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################


def parse_args(args: list[str]):
    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Run mode: train_forward, train_reverse, train_bidirectional, predict_forward, predict_reverse, predict_bidirectional.",
    )
    parser.add_argument(
        "--in_file",
        type=str,
        required=True,
        help="Path to input YAML configuration file.",
    )
    parser.add_argument(
        "--in_model",
        type=str,
        help="Path to a pre-trained model directory (optional).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the results to disk.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow logging and save logs to disk.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging and save logs to disk.",
    )

    args = parser.parse_args(args)
    return args


###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(args: list[str]):
    # Parsing command line arguments
    args = parse_args(args)

    logging.info(f"Loading YAML configuration file @ {args.in_file}")
    with open(args.in_file, "r") as f:
        config = yaml.safe_load(f)

    # Branching into training or prediction mode
    if "train" in args.mode:
        train(config, args)  # Run training
    elif "predict" in args.mode:
        metadata_file = os.path.join(args.in_model, "metadata.yaml")
        try:
            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f)
        except:
            raise ValueError(f"Something is wrong with your YAML metadata file @ {metadata_file}.")
        predict(config, args, metadata)  # Run prediction
    else:
        raise ValueError(f"Incorrect mode: {args.mode}.")


################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################


if __name__ == "__main__":
    main(sys.argv[1:])
