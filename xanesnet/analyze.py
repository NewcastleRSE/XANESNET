# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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
#
# Citations:
#   ...

"""Entry point and argument parser for analysis runs."""

import logging
from argparse import ArgumentParser, Namespace

from xanesnet.analysis.aggregators import AggregatorRegistry
from xanesnet.analysis.collectors import CollectorRegistry
from xanesnet.analysis.plotters import PlotterRegistry
from xanesnet.analysis.reporters import ReporterRegistry
from xanesnet.analysis.selectors import SelectorRegistry
from xanesnet.core_analyze import analyze
from xanesnet.losses import LossRegistry
from xanesnet.serialization.config import (
    Config,
    ConfigRaw,
    copy_raw_config,
    load_raw_config,
)
from xanesnet.serialization.metadata import write_run_metadata
from xanesnet.serialization.schema_validation import validate_config_schema
from xanesnet.utils.filesystem import create_run_dir, create_subfolders
from xanesnet.utils.logger import setup_file_logging, setup_logging
from xanesnet.utils.prompts import auto_yes
from xanesnet.utils.random import set_global_seed

###############################################################################
################################### LOGGING ###################################
###############################################################################

setup_logging(logging.INFO)

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################


def _flatten_args(values: list[list[str]] | None) -> list[str]:
    """Flatten the nested lists produced by ``nargs='+'`` with ``action='append'``.

    Args:
        values: Nested list of parsed values, or ``None``.

    Returns:
        Flat list of values in argument order.
    """
    if not values:
        return []
    return [value for group in values for value in group]


def parse_args(args: list[str]) -> Namespace:
    """Parse command-line arguments for the analysis entry point.

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
        "-r",
        "--inference-runs",
        type=str,
        required=True,
        action="append",
        nargs="+",
        help=("Path(s) to inference run directories. Can be repeated (-r A -r B) " "and/or space separated (-r A B)."),
    )
    parser.add_argument(
        "-d",
        "--prediction-names",
        type=str,
        action="append",
        nargs="+",
        help=(
            "Optional display names for inference run directories, in the same "
            "order as -r. Can be repeated (-d X -d Y) and/or space separated "
            "(-d X Y)."
        ),
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
        help="Optional name for the analysis run.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer yes to confirmation prompts.",
    )

    args_namespace = parser.parse_args(args)

    args_namespace.inference_runs = _flatten_args(args_namespace.inference_runs)
    args_namespace.prediction_names = _flatten_args(args_namespace.prediction_names)

    return args_namespace


###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(args: list[str]) -> None:
    """Run the full analysis pipeline.

    Parses arguments, configures prompt behavior, loads configuration, sets up
    the run directory, and delegates to the ``analyze`` core function.

    Args:
        args: Raw command-line argument strings.
    """
    # Registry printing
    logging.debug("REGISTRY:")
    logging.debug(f"\tSelectors: {SelectorRegistry.list()}")
    logging.debug(f"\tCollectors: {CollectorRegistry.list()}")
    logging.debug(f"\tAggregators: {AggregatorRegistry.list()}")
    logging.debug(f"\tPlotters: {PlotterRegistry.list()}")
    logging.debug(f"\tReporters: {ReporterRegistry.list()}")
    logging.debug(f"\tLosses: {LossRegistry.list()}")

    # Parsing command line arguments
    args_namespace = parse_args(args)

    with auto_yes(args_namespace.yes):
        # Loading configuration file
        logging.info(f"Loading YAML configuration file @ {args_namespace.in_file}")
        config_raw: ConfigRaw = load_raw_config(args_namespace.in_file)

        # Get saving directory
        out_dir = "./runs" if args_namespace.out_dir is None else args_namespace.out_dir
        save_dir = create_run_dir(out_dir, name=f"analyze_{args_namespace.name}" if args_namespace.name else "analyze")
        logging.info(f"Run directory: {save_dir}")
        create_subfolders(save_dir, subfolder_names=["plots", "reports", "aux"])

        # Setup file logging
        setup_file_logging(save_dir)

        # Write software and hardware metadata files
        software_info_path, hardware_info_path = write_run_metadata(save_dir, mode="analyze", command_line_args=args)
        logging.info(f"Software metadata saved to: {software_info_path}")
        logging.info(f"Hardware metadata saved to: {hardware_info_path}")

        # Copy raw config file
        config_save_path = copy_raw_config(args_namespace.in_file, save_dir, new_name="analyze_config.yaml")
        logging.info(f"Configuration file saved to: {config_save_path}")

        # Config validation
        validated = validate_config_schema(config_raw, "analyze")
        config: Config = Config(validated)
        validate_config_save_path = config.save(save_dir / "validated_analyze_config.yaml")
        logging.info(f"Validated config file saved to: {validate_config_save_path}.")

        # Setting global seed for reproducibility
        seed = config.get_optional_int("seed")
        if seed is None:
            logging.warning("No global seed specified in configuration file. Choosing random seed.")
        seed = set_global_seed(seed)
        logging.info(f"Global seed: {seed}")

        # Branching into analyze mode
        analyze(config, args_namespace, save_dir)
