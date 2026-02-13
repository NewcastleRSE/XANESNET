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

from xanesnet.analysis.aggregators import AggregatorRegistry
from xanesnet.analysis.collectors import CollectorRegistry
from xanesnet.analysis.plotters import PlotterRegistry
from xanesnet.analysis.reporters import ReporterRegistry
from xanesnet.analysis.selectors import SelectorRegistry
from xanesnet.core_analyze import analyze
from xanesnet.serialization.config import (
    Config,
    ConfigRaw,
    copy_raw_config,
    load_raw_config,
    validate_config_analyze,
)
from xanesnet.utils.filesystem import create_run_dir, create_subfolders
from xanesnet.utils.logger import setup_file_logging, setup_logging
from xanesnet.utils.random import set_global_seed

###############################################################################
################################### LOGGING ###################################
###############################################################################

setup_logging(logging.INFO)

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
        "-p",
        "--predictions",
        type=str,
        required=True,
        help="Path to directory containing predictions.Can be specified multiple times.",
        action="append",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Optional name for the analysis run.",
    )

    args_namespace = parser.parse_args(args)
    return args_namespace


###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(args: list[str]) -> None:
    # Registry printing
    logging.debug("REGISTRY:")
    logging.debug(f"\tSelectors: {SelectorRegistry.list()}")
    logging.debug(f"\tCollectors: {CollectorRegistry.list()}")
    logging.debug(f"\tAggregators: {AggregatorRegistry.list()}")
    logging.debug(f"\tPlotters: {PlotterRegistry.list()}")
    logging.debug(f"\tReporters: {ReporterRegistry.list()}")

    # Parsing command line arguments
    args_namespace = parse_args(args)

    # Loading configuration file
    logging.info(f"Loading YAML configuration file @ {args_namespace.in_file}")
    config_raw: ConfigRaw = load_raw_config(args_namespace.in_file)

    # Get saving directory
    save_dir = create_run_dir("./runs", name=f"analyze_{args_namespace.name}" if args_namespace.name else "analyze")
    logging.info(f"Run directory: {save_dir}")
    create_subfolders(save_dir, subfolder_names=["plots", "reports", "aux"])

    # Setup file logging
    setup_file_logging(save_dir)

    # Copy raw config file
    config_save_path = copy_raw_config(args_namespace.in_file, save_dir, new_name="analyze_config.yaml")
    logging.info(f"Configuration file saved to: {config_save_path}")

    # Config validation
    config: Config = validate_config_analyze(config_raw)
    validate_config_save_path = config.save(save_dir / "validated_analyze_config.yaml")
    logging.info(f"Validated config file saved to: {validate_config_save_path}.")

    # Setting global seed for reproducibility
    seed = config.get_optional_int("seed")
    if seed is None:
        logging.warning("No global seed specified in configuration file. Choosing random seed.")
    seed = set_global_seed(seed)
    logging.info(f"Global seed: {seed}")

    # Branching into analyze mode
    analyze(config, args_namespace, save_dir)  # Run analysis
