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
from typing import Dict, List, Tuple

import torch
from torchinfo import summary

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.datasets import Dataset, DatasetRegistry
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.models import Model
from xanesnet.strategies import Strategy, StrategyRegistry
from xanesnet.utils.io import save_models
from xanesnet.utils.mode import Mode, get_mode

###############################################################################
#################################### TRAIN ####################################
###############################################################################


def train(config, args):
    """
    Main training entry
    """
    logging.info(f"Training mode: {args.mode}")
    mode = get_mode(args.mode)

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(config, mode, datasource)
    strategy = _setup_strategy(config, dataset)
    strategy.setup_models()
    strategy.setup_learners()

    # Main training
    model_list, train_time = _run_training(strategy)

    # Display model summary and training duration
    logging.info(f"Number of trained models: {len(model_list)}")
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")
    _summary_models(model_list, dataset)

    # Save model(s) and metadata to disk if requested
    # TODO check metadata saving !!!
    if args.save:
        metadata = {
            "mode": args.mode,
            "dataset": dataset.config,
            "model": None,  # TODO
            "scheme": None,  # TODO
        }
        save_models(Path("models"), model_list, metadata, dataset.basis)  # TODO


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
    logging.info(f"Initialising training dataset: {dataset_type}")
    dataset = DatasetRegistry.get(dataset_type)(**config["dataset"], mode=mode, datasource=datasource)
    dataset.process()
    dataset.check_preload()  # may preload the dataset into memory

    # Log dataset summary
    logging.info(f"Dataset Summary: # of samples = {len(dataset)}")

    return dataset


def _setup_strategy(config: Dict, dataset: Dataset) -> Strategy:
    """
    Initialises the learning strategy.
    """
    strategy_config = config["strategy"]
    strategy_type = strategy_config["strategy_type"]

    model_config = config["model"]
    learner_config = config["learner"]

    logging.info(f"Initialising strategy: {strategy_type}")
    strategy = StrategyRegistry.get(strategy_type)(
        **strategy_config,
        dataset=dataset,
        model_config=model_config,
        learner_config=learner_config,
    )

    return strategy


###############################################################################
############################## TRAINING STARTER ###############################
###############################################################################


def _run_training(strategy: Strategy) -> Tuple[List, float]:
    """
    Train using the selected training scheme.
    """
    start_time = time.time()

    model_list = strategy.run_training()

    train_time = time.time() - start_time

    # Move to CPU
    for model in model_list:
        model.to(torch.device("cpu"))

    return model_list, train_time


###############################################################################
############################### SUMMARY LOGGING ###############################
###############################################################################


def _summary_models(model_list: List[Model], dataset: Dataset) -> None:
    logging.info("Model Summary")

    for idx, model in enumerate(model_list):
        batchprocessor = BatchProcessorRegistry.get(dataset.dataset_type, model.model_type)
        sample = dataset[0]
        inputs = batchprocessor.input_preparation_single(sample)
        logging.info(f"Model  {idx}:")
        summary(model, input_data=inputs)
        logging.info("")
