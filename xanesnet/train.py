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

"""Entry point and argument parser for training runs."""

import logging
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.core_train import train
from xanesnet.datasets import DatasetRegistry
from xanesnet.datasources import DataSourceRegistry
from xanesnet.descriptors import DescriptorRegistry
from xanesnet.models import ModelRegistry
from xanesnet.runners.trainers import TrainerRegistry
from xanesnet.serialization.config import (
    Config,
    ConfigRaw,
    copy_raw_config,
    load_raw_config,
    validate_config_train,
)
from xanesnet.serialization.tensorboard import tb_logger
from xanesnet.strategies import StrategyRegistry
from xanesnet.utils.filesystem import create_run_dir, create_subfolders
from xanesnet.utils.logger import setup_file_logging, setup_logging
from xanesnet.utils.random import set_global_seed

###############################################################################
################################### LOGGING ###################################
###############################################################################

setup_logging(logging.DEBUG)

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################


def parse_args(args: list[str]) -> Namespace:
    """Parse command-line arguments for the training entry point.

    Args:
        args: Raw command-line argument strings.

    Returns:
        Parsed argument namespace.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_file",
        type=str,
        required=True,
        help="Path to input YAML configuration file.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Path to output directory. (Optional, default: ./runs )",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Optional name for the training run.",
    )
    parser.add_argument(
        "-t",
        "--tensorboard",
        action="store_true",
        help="Whether to write training metrics to TensorBoard logs.",
    )

    args_namespace = parser.parse_args(args)
    return args_namespace


###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(args: list[str]) -> None:
    """Run the full training pipeline.

    Parses arguments, loads and validates configuration, sets up the run
    directory, and delegates to the ``train`` core function.

    Args:
        args: Raw command-line argument strings.
    """
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
    args_namespace = parse_args(args)

    # Loading configuration file
    logging.info(f"Loading YAML configuration file @ {args_namespace.in_file}")
    config_raw: ConfigRaw = load_raw_config(args_namespace.in_file)

    # Get saving directory
    out_dir = "./runs" if args_namespace.out_dir is None else args_namespace.out_dir
    save_dir = create_run_dir(out_dir, name=f"train_{args_namespace.name}" if args_namespace.name else "train")
    logging.info(f"Run directory: {save_dir}")
    subfolders = ["models", "checkpoints"] + (["tensorboard"] if args_namespace.tensorboard else [])
    create_subfolders(save_dir, subfolder_names=subfolders)

    # Setup file logging
    setup_file_logging(save_dir)

    # Copy raw config file
    config_save_path = copy_raw_config(args_namespace.in_file, save_dir, new_name="train_config.yaml")
    logging.info(f"Configuration file saved to: {config_save_path}")

    # Config validation
    config: Config = validate_config_train(config_raw)
    validate_config_save_path = config.save(save_dir / "validated_train_config.yaml")
    logging.info(f"Validated config file saved to: {validate_config_save_path}.")

    # Scale file copying (if configured)
    scale_file = config.section("model").as_kwargs().get("scale_file")
    if scale_file:
        src = Path(scale_file)
        if src.exists():
            dst = save_dir / "models" / "scale_factors.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            logging.info(f"Copied model.scale_file {src} -> {dst}")
        else:
            logging.warning(f"Configured model.scale_file does not exist on disk: {src}")

    # Setting global seed for reproducibility
    seed = config.get_optional_int("seed")
    if seed is None:
        logging.warning("No global seed specified in configuration file. Choosing random seed.")
    seed = set_global_seed(seed)
    logging.info(f"Global seed: {seed}")

    # Tensorboard
    if args_namespace.tensorboard:
        logging.info("TensorBoard logging enabled.")
        tb_logger.set_config(config)
    else:
        logging.info("TensorBoard logging disabled.")

    # Branching into training mode
    train(config, args_namespace, save_dir)
