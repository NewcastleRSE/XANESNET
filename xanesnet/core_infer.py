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

from xanesnet.datasets import Dataset, DatasetRegistry
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config
from xanesnet.strategies import Strategy, StrategyRegistry

###############################################################################
#################################### INFER ####################################
###############################################################################


def infer(config: Config, args_namespace: Namespace, save_dir: Path, checkpoint: Checkpoint) -> None:
    """
    Main inference entry
    """
    logging.info(f"Inference from checkpoint.")

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(config, datasource)
    strategy = _setup_strategy(config, dataset)
    strategy.setup_models()
    strategy.set_state_dicts(checkpoint.model_states)
    strategy.setup_inferencers(config.get_str("device"))

    predictions_save_path: Path | None = save_dir / "predictions"

    # Main inference
    inference_time = _run_inference(strategy, predictions_save_path)

    # Summary
    logging.info(f"Inference completed in {str(timedelta(seconds=int(inference_time)))}")


###############################################################################
############################### SETUP FUNCTIONS ###############################
###############################################################################


def _setup_datasource(config: Config) -> DataSource:
    """
    Setup the data source from config
    """
    datasource_config = config.section("datasource")
    datasource_type = datasource_config.get_str("datasource_type")
    logging.info(f"Initialising data source: {datasource_type}")
    datasource = DataSourceRegistry.get(datasource_type)(**datasource_config.as_kwargs())

    return datasource


def _setup_dataset(config: Config, datasource: DataSource) -> Dataset:
    """
    Process the dataset using input configuration or load an existing one from disk
    """
    dataset_config = config.section("dataset")
    dataset_type = dataset_config.get_str("dataset_type")

    logging.info(f"Initialising inference dataset: {dataset_type}")
    dataset = DatasetRegistry.get(dataset_type)(**dataset_config.as_kwargs(), datasource=datasource)
    dataset.prepare()
    dataset.check_preload()  # may preload the dataset into memory

    # Log dataset summary
    logging.info(f"Dataset Summary: # of samples = {len(dataset)}")

    return dataset


def _setup_strategy(config: Config, dataset: Dataset) -> Strategy:
    strategy_config = config.section("strategy")
    strategy_type = strategy_config.get_str("strategy_type")

    model_config = config.section("model")
    inferencer_config = config.section("inferencer")

    logging.info(f"Initialising strategy: {strategy_type}")
    strategy = StrategyRegistry.get(strategy_type)(
        **strategy_config.as_kwargs(),
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
