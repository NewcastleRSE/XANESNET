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
from datetime import timedelta
from pathlib import Path
from typing import Dict

from xanesnet.datasets import Dataset, DatasetRegistry
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.strategies import Strategy, StrategyRegistry
from xanesnet.utils.io import Checkpoint, copy_yaml
from xanesnet.utils.mode import Mode, get_mode

###############################################################################
#################################### INFER ####################################
###############################################################################


def infer(config, args, save_dir: Path, checkpoint: Checkpoint):
    """
    Main inference entry
    """
    mode = get_mode(config["mode"])
    logging.info(f"Inference mode from checkpoint: {mode}")

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(config, mode, datasource)
    strategy = _setup_strategy(config, dataset)
    strategy.setup_models()
    strategy.set_state_dicts(checkpoint.model_states)
    strategy.setup_inferencers(config.get("device", "cpu"))  # TODO

    # Save inference config
    if args.save:
        config_save_path = copy_yaml(args.in_file, save_dir, new_name="infer_config.yaml")
        logging.info(f"Configuration file saved to: {config_save_path}")

    # Main inference
    _, inference_time = _run_inference(strategy)  # TODO

    # Summary
    logging.info(f"Inference completed in {str(timedelta(seconds=int(inference_time)))}")
    # TODO add inference summary

    # Saving
    if args.save:
        pass
        # TODO save inference results


###############################################################################
############################### SETUP FUNCTIONS ###############################
###############################################################################


def _setup_datasource(config: Dict) -> DataSource:
    """
    Setup the data source from config
    """
    datasource_type = config[f"datasource"]["datasource_type"]
    logging.info(f"Initialising data source: {datasource_type}")
    datasource = DataSourceRegistry.get(datasource_type)(**config["datasource"])

    return datasource


def _setup_dataset(config: Dict, mode: Mode, datasource: DataSource) -> Dataset:
    """
    Process the dataset using input configuration or load an existing one from disk
    """
    dataset_type = config["dataset"]["dataset_type"]
    logging.info(f"Initialising inference dataset: {dataset_type}")
    dataset = DatasetRegistry.get(dataset_type)(**config["dataset"], mode=mode, datasource=datasource)
    dataset.process()
    dataset.check_preload()  # may preload the dataset into memory

    # Log dataset summary
    logging.info(f"Dataset Summary: # of samples = {len(dataset)}")

    return dataset


def _setup_strategy(config: Dict, dataset: Dataset) -> Strategy:
    strategy_config = config["strategy"]
    strategy_type = strategy_config["strategy_type"]

    model_config = config["model"]
    inferencer_config = config["inferencer"]

    logging.info(f"Initialising strategy: {strategy_type}")
    strategy = StrategyRegistry.get(strategy_type)(
        **strategy_config,
        dataset=dataset,
        model_config=model_config,
        inferencer_config=inferencer_config,
    )

    return strategy


###############################################################################
############################## INFERENCE STARTER ##############################
###############################################################################


def _run_inference(strategy: Strategy):
    """
    Run inference using the selected inference strategy.
    """
    start_time = time.time()

    # TODO run inference
    strategy.run_inference()

    inference_time = time.time() - start_time

    return None, inference_time  # TODO ?
