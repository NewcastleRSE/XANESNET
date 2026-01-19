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
from argparse import ArgumentParser

import yaml

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.core_train import train
from xanesnet.datasets import DatasetRegistry
from xanesnet.datasources import DataSourceRegistry
from xanesnet.descriptors import DescriptorRegistry
from xanesnet.models import ModelRegistry
from xanesnet.strategies import StrategyRegistry
from xanesnet.trainers import TrainerRegistry
from xanesnet.utils.io import create_run_dir, create_subfolders
from xanesnet.utils.logger import setup_file_logging, setup_logging
from xanesnet.utils.random import set_global_seed

###############################################################################
################################### LOGGING ###################################
###############################################################################

setup_logging(logging.DEBUG)

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################


def parse_args(args: list[str]):
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_file",
        type=str,
        required=True,
        help="Path to input YAML configuration file.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the results to disk.",
    )

    # TODO these arguments are not tested yet !!!
    parser.add_argument(
        "--in_model",
        type=str,
        help="Path to a pre-trained model directory (optional).",
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
    # Registry printing
    logging.debug("REGISTRY:")
    logging.debug(f"\tData Sources: {DataSourceRegistry.list()}")
    logging.debug(f"\tDatasets: {DatasetRegistry.list()}")
    logging.debug(f"\tDescriptors: {DescriptorRegistry.list()}")
    logging.debug(f"\tModels: {ModelRegistry.list()}")
    logging.debug(f"\tTrainers: {TrainerRegistry.list()}")
    logging.debug(f"\tBatchProcessers: {BatchProcessorRegistry.list()}")
    logging.debug(f"\tStrategies: {StrategyRegistry.list()}")

    # Parsing command line arguments
    args = parse_args(args)

    # Loading configuration file
    logging.info(f"Loading YAML configuration file @ {args.in_file}")
    with open(args.in_file, "r") as f:
        config = yaml.safe_load(f)

    # Get saving directory
    save_dir = create_run_dir("./runs", name=f"{config["model"]["model_type"]}_{config["strategy"]["strategy_type"]}")
    logging.info(f"Run directory: {save_dir}")
    create_subfolders(save_dir, subfolder_names=["models", "plots", "checkpoints"])

    # Setup file logging
    setup_file_logging(save_dir)

    # Setting global seed for reproducibility
    seed = config.get("seed", None)
    if seed is None:
        logging.warning("No global seed specified in configuration file. Choosing random seed.")
    seed = set_global_seed(seed)
    logging.info(f"Global seed: {seed}")

    # Branching into training mode
    train(config, args, save_dir)  # Run training
