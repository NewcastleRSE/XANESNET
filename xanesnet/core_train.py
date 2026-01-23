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

import torch
from torchinfo import summary

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.datasets import Dataset, DatasetRegistry
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.models import Model
from xanesnet.serialization import (
    build_checkpoint,
    save_checkpoint,
    save_dict_as_yaml,
    save_models,
)
from xanesnet.strategies import Strategy, StrategyRegistry
from xanesnet.utils import copy_file

###############################################################################
#################################### TRAIN ####################################
###############################################################################


def train(config: dict[str, Any], args_namespace: Namespace, save_dir: Path) -> None:
    """
    Main training entry
    """
    logging.info(f"Training.")

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(config, datasource)
    strategy = _setup_strategy(config, dataset, save_dir / "checkpoints")
    strategy.setup_models()
    strategy.setup_checkpointer()
    strategy.init_model_weights()
    strategy.setup_trainers(config["device"])

    # Save training config and signature
    signature = None
    if args_namespace.save:
        config_save_path = copy_file(args_namespace.in_file, save_dir, new_name="train_config.yaml")
        logging.info(f"Configuration file saved to: {config_save_path}")
        signature = {
            "dataset": dataset.signature,
            "model": strategy.model_signature,
            "strategy": strategy.signature,
        }
        signature_save_path = save_dict_as_yaml(signature, save_dir / "models", "signature")
        logging.info(f"Signature saved to: {signature_save_path}")

    # Main training
    model_list, train_time = _run_training(strategy)

    # Display model summary and training duration
    logging.info(f"Number of trained models: {len(model_list)}")
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")
    _summary_models(model_list, dataset)

    # Save model(s)
    if args_namespace.save:
        save_models(save_dir / "models", model_list)
        logging.info(f"Trained model(s) saved to: {save_dir / 'models'}")
        assert signature is not None
        final_checkpoint = build_checkpoint(model_list, signature=signature)
        final_save_path = save_checkpoint(save_dir / "models", final_checkpoint, name="final")
        logging.info(f"Final checkpoint without optimizers and epochs saved @ {final_save_path}")


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

    logging.info(f"Initialising training dataset: {dataset_type}")
    dataset = DatasetRegistry.get(dataset_type)(**dataset_config, datasource=datasource)
    dataset.process()
    dataset.check_preload()  # may preload the dataset into memory

    # Log dataset summary
    logging.info(f"Dataset Summary: # of samples = {len(dataset)}")

    return dataset


def _setup_strategy(config: dict[str, Any], dataset: Dataset, checkpoint_dir: str | Path) -> Strategy:
    """
    Initialises the training strategy.
    """
    strategy_config = config["strategy"]
    strategy_type = strategy_config["strategy_type"]

    model_config = config["model"]
    trainer_config = config["trainer"]

    logging.info(f"Initialising strategy: {strategy_type}")
    strategy = StrategyRegistry.get(strategy_type)(
        **strategy_config,
        checkpoint_dir=checkpoint_dir,
        dataset=dataset,
        model_config=model_config,
        trainer_config=trainer_config,
    )

    return strategy


###############################################################################
############################## TRAINING STARTER ###############################
###############################################################################


def _run_training(strategy: Strategy) -> tuple[list[Model], float]:
    """
    Train using the selected training strategy.
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


def _summary_models(model_list: list[Model], dataset: Dataset) -> None:
    logging.info("Model Summary")

    for idx, model in enumerate(model_list):
        batchprocessor = BatchProcessorRegistry.get(dataset.dataset_type, model.model_type)()
        sample = dataset[0]
        inputs = batchprocessor.input_preparation_single(sample)
        logging.info(f"Model  {idx}:")
        summary(model, input_data=inputs)
