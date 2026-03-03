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

import torch
from torchinfo import summary

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.datasets import Dataset, DatasetRegistry
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.models import Model
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config
from xanesnet.serialization.models import save_models
from xanesnet.serialization.splits import save_split_indices
from xanesnet.strategies import Strategy, StrategyRegistry

###############################################################################
#################################### TRAIN ####################################
###############################################################################


def train(config: Config, args_namespace: Namespace, save_dir: Path) -> None:
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
    strategy.setup_trainers(config.get_str("device"))

    # Save signature
    signature = Config(
        {
            "dataset": dataset.signature,
            "model": strategy.model_signature,
            "strategy": strategy.signature,
        }
    )
    signature_save_path = signature.save(save_dir / "models" / "signature.yaml")
    logging.info(f"Signature saved to: {signature_save_path}")

    # Save split indices if they were generated
    split_indices_save_path = save_dir / "split_indices.json"
    save_split_indices(split_indices_save_path, dataset.get_all_subset_indices())
    logging.info(f"Split indices saved to: {split_indices_save_path}")

    # Main training
    model_list, train_time = _run_training(strategy)

    # Display model summary and training duration
    logging.info(f"Number of trained models: {len(model_list)}")
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")
    _summary_models(model_list, dataset)

    # Save model(s)
    save_models(save_dir / "models", model_list)
    logging.info(f"Trained model(s) saved to: {save_dir / 'models'}")
    final_checkpoint = Checkpoint.build(model_list, signature=signature)
    final_save_path = final_checkpoint.save(save_dir / "models" / "final.pth")
    logging.info(f"Final checkpoint without optimizers and epochs saved @ {final_save_path}")


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

    logging.info(f"Initialising training dataset: {dataset_type}")
    dataset = DatasetRegistry.get(dataset_type)(**dataset_config.as_kwargs(), datasource=datasource)
    dataset.prepare()
    dataset.check_preload()  # may preload the dataset into memory

    # Log dataset summary
    logging.info(f"Dataset Summary: # of samples = {len(dataset)}")

    return dataset


def _setup_strategy(config: Config, dataset: Dataset, checkpoint_dir: str | Path) -> Strategy:
    """
    Initialises the training strategy.
    """
    strategy_config = config.section("strategy")
    strategy_type = strategy_config.get_str("strategy_type")

    model_config = config.section("model")
    trainer_config = config.section("trainer")

    logging.info(f"Initialising strategy: {strategy_type}")
    strategy = StrategyRegistry.get(strategy_type)(
        **strategy_config.as_kwargs(),
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
        inputs = batchprocessor.input_preparation_single(dataset, 0)
        logging.info(f"Model  {idx}:")
        summary(model, input_data=inputs)
