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

import yaml

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.core_infer import infer
from xanesnet.datasets import DatasetRegistry
from xanesnet.datasources import DataSourceRegistry
from xanesnet.descriptors import DescriptorRegistry
from xanesnet.models import ModelRegistry
from xanesnet.runners.trainers import TrainerRegistry
from xanesnet.strategies import StrategyRegistry
from xanesnet.utils import set_global_seed, setup_file_logging, setup_logging
from xanesnet.utils.io import (
    Checkpoint,
    create_run_dir,
    create_subfolders,
    merge_configs,
    save_dict_as_yaml,
    validate_config,
)

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
        help="Path to a trained model .pth file.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the results to disk.",
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
    with open(args_namespace.in_file, "r") as f:
        config = yaml.safe_load(f)

    # Loading model/signature for inference
    checkpoint = Checkpoint.load(args_namespace.in_model)

    # Get saving directory
    save_dir = create_run_dir(
        "./runs",
        name=f"infer_{checkpoint.signature['model']['model_type']}_{checkpoint.signature['strategy']['strategy_type']}",
    )
    logging.info(f"Run directory: {save_dir}")
    create_subfolders(save_dir, subfolder_names=["plots"])

    # Setup file logging
    setup_file_logging(save_dir)

    # Merge inference config and checkpoint config
    config = merge_configs(config, checkpoint.signature)
    merged_config_save_path = save_dict_as_yaml(config, save_dir, "merged_infer_config")
    config["dataset"].pop("mode", None)  # Remove redundant mode from dataset config
    logging.info(f"Merged configuration file saved to: {merged_config_save_path}.")

    # Config validation
    config = validate_config(config)
    validate_config_save_path = save_dict_as_yaml(config, save_dir, "validated_config")
    logging.info(f"Validated config file saved to: {validate_config_save_path}.")

    # Setting global seed for reproducibility
    seed = config["seed"]
    if seed is None:
        logging.warning("No global seed specified in configuration file. Choosing random seed.")
    seed = set_global_seed(seed)
    logging.info(f"Global seed: {seed}")

    # Branching into inference mode
    infer(config, args_namespace, save_dir, checkpoint)  # Run inference
