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
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from typing import Any

from xanesnet.datasets import Dataset, DatasetRegistry
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.serialization import Checkpoint
from xanesnet.strategies import Strategy, StrategyRegistry
from xanesnet.utils import copy_file

# TODO better inference:
# TODO logging metrics, saving plots, saving results, etc.
# TODO What do i need to save from inference to be useful?
# TODO What to report:
# TODO - predictions vs ground truth
# TODO - metrics (per sample, overall, energy resolved, different loss functions, etc.)
# TODO - -> mean, median, std, min, max, percentiles
# TODO - inference time (total, per sample, per batch, etc.)
# TODO - Model parameters count
# TODO - performance stratified by different parameters
# TODO - PLOTS:
# TODO - * error distribution plots/histograms
# TODO - * averaged error vs energy plots
# TODO - * predicted vs ground truth vs error metrics plots
# TODO
# TODO SPLIT THIS INTO INFERENCE AND RESULTS ANALYSIS -> Keep inference light,
# TODO and have a separate analysis script that loads results and does all
# TODO the analysis and plotting.


###############################################################################
#################################### INFER ####################################
###############################################################################


def infer(config: dict[str, Any], args_namespace: Namespace, save_dir: Path, checkpoint: Checkpoint) -> None:
    """
    Main inference entry
    """
    logging.info(f"Inference from checkpoint.")

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(config, datasource)
    strategy = _setup_strategy(config, dataset)
    strategy.setup_models()
    strategy.set_state_dicts(checkpoint.model_states)
    strategy.setup_inferencers(config["device"])

    # Save inference config
    predictions_save_path: Path | None = None
    if args_namespace.save:
        config_save_path = copy_file(args_namespace.in_file, save_dir, new_name="infer_config.yaml")
        logging.info(f"Configuration file saved to: {config_save_path}")

        predictions_save_path = save_dir / "predictions"

    # Main inference
    inference_time = _run_inference(strategy, predictions_save_path)

    # Summary
    logging.info(f"Inference completed in {str(timedelta(seconds=int(inference_time)))}")
    # TODO add inference summary

    # Saving
    if args_namespace.save:
        pass
        # TODO save inference results


###############################################################################
############################### SETUP FUNCTIONS ###############################
###############################################################################


def _setup_datasource(config: dict[str, Any]) -> DataSource:
    """
    Setup the data source from config
    """
    datasource_type = config[f"datasource"]["datasource_type"]
    logging.info(f"Initialising data source: {datasource_type}")
    datasource = DataSourceRegistry.get(datasource_type)(**config["datasource"])

    return datasource


def _setup_dataset(config: dict[str, Any], datasource: DataSource) -> Dataset:
    """
    Process the dataset using input configuration or load an existing one from disk
    """
    dataset_config = config["dataset"]
    dataset_type = dataset_config["dataset_type"]

    logging.info(f"Initialising inference dataset: {dataset_type}")
    dataset = DatasetRegistry.get(dataset_type)(**dataset_config, datasource=datasource)
    dataset.prepare()
    dataset.check_preload()  # may preload the dataset into memory

    # Log dataset summary
    logging.info(f"Dataset Summary: # of samples = {len(dataset)}")

    return dataset


def _setup_strategy(config: dict[str, Any], dataset: Dataset) -> Strategy:
    strategy_config = config["strategy"]
    strategy_type = strategy_config["strategy_type"]

    model_config = config["model"]
    inferencer_config = config["inferencer"]

    logging.info(f"Initialising strategy: {strategy_type}")
    strategy = StrategyRegistry.get(strategy_type)(
        **strategy_config,
        checkpoint_dir=None,
        dataset=dataset,
        model_config=model_config,
        inferencer_config=inferencer_config,
    )

    return strategy


###############################################################################
############################## INFERENCE STARTER ##############################
###############################################################################


def _run_inference(strategy: Strategy, predictions_save_path: str | Path | None) -> float:
    """
    Run inference using the selected inference strategy.
    """
    start_time = time.time()

    strategy.run_inference(predictions_save_path)

    inference_time = time.time() - start_time

    return inference_time
