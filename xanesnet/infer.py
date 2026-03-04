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
from argparse import ArgumentParser, Namespace

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.core_infer import infer
from xanesnet.datasets import DatasetRegistry
from xanesnet.datasources import DataSourceRegistry
from xanesnet.descriptors import DescriptorRegistry
from xanesnet.models import ModelRegistry
from xanesnet.runners.trainers import TrainerRegistry
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import (
    Config,
    ConfigRaw,
    copy_raw_config,
    load_raw_config,
    merge_raw_configs,
    save_raw_config,
    validate_config_infer,
)
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
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_file",
        type=str,
        required=True,
        help="Path to input YAML configuration file.",
    )
    parser.add_argument(
        "-m",
        "--in_model",
        type=str,
        required=True,
        help="Path to a trained model .pth file.",
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
        help="Optional name for the inference run.",
    )

    args_namespace = parser.parse_args(args)
    return args_namespace


###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(args: list[str]) -> None:
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

    # Loading model/signature for inference
    logging.info(f"Loading trained model checkpoint @ {args_namespace.in_model}")
    checkpoint = Checkpoint.load(args_namespace.in_model)

    # Get saving directory
    out_dir = "./runs" if args_namespace.out_dir is None else args_namespace.out_dir
    save_dir = create_run_dir(out_dir, name=f"infer_{args_namespace.name}" if args_namespace.name else "infer")
    logging.info(f"Run directory: {save_dir}")
    create_subfolders(save_dir, subfolder_names=["predictions"])

    # Setup file logging
    setup_file_logging(save_dir)

    # Copy raw config file
    config_save_path = copy_raw_config(args_namespace.in_file, save_dir, new_name="infer_config.yaml")
    logging.info(f"Configuration file saved to: {config_save_path}")

    # Merge inference config and checkpoint config
    config_raw = merge_raw_configs(config_raw, checkpoint.signature.as_dict())
    merged_config_save_path = save_raw_config(config_raw, save_dir / "merged_infer_config.yaml")
    logging.info(f"Merged configuration file saved to: {merged_config_save_path}.")

    # Config validation
    config: Config = validate_config_infer(config_raw)
    validate_config_save_path = config.save(save_dir / "validated_infer_config.yaml")
    logging.info(f"Validated config file saved to: {validate_config_save_path}.")

    # Setting global seed for reproducibility
    seed = config.get_optional_int("seed")
    if seed is None:
        logging.warning("No global seed specified in configuration file. Choosing random seed.")
    seed = set_global_seed(seed)
    logging.info(f"Global seed: {seed}")

    # Branching into inference mode
    infer(config, args_namespace, save_dir, checkpoint)  # Run inference
