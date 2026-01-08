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

from xanesnet.datasets.dataset import Dataset
from xanesnet.datasources import DataSource
from xanesnet.models.base_model import Model
from xanesnet.models.pre_trained import PretrainedModels
from xanesnet.registry import (
    DatasetRegistry,
    DataSourceRegistry,
    ModelRegistry,
    SchemeRegistry,
)
from xanesnet.scheme import Learn
from xanesnet.utils.io import load_pretrained_model, save_models
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
    model = _setup_model(config, dataset)
    scheme = _setup_scheme(config, args, model, dataset)

    # Main training
    model_list, scheme_type, train_time = _run_training(config, scheme)

    # Display model summary and training duration
    _summary_model(model_list[0], dataset)
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")

    # Save model(s) and metadata to disk if requested
    if args.save:
        metadata = {
            "mode": args.mode,
            "dataset": dataset.config,
            "model": model.config,
            "scheme": scheme_type,
        }
        save_models(Path("models"), model_list, metadata, dataset.basis)


###############################################################################
############################### SETUP FUNCTIONS ###############################
###############################################################################


def _setup_datasource(config: Dict) -> DataSource:
    """
    Setup the data source from config
    """
    datasource_type = config[f"datasource"]["type"]
    logging.info(f"Initialising data source: {datasource_type}")
    datasource = DataSourceRegistry.get(datasource_type)(**config["datasource"])

    return datasource


def _setup_dataset(config: Dict, mode: Mode, datasource: DataSource) -> Dataset:
    """
    Process the dataset using input configuration or load an existing one from disk
    """
    dataset_type = config["dataset"]["type"]
    logging.info(f"Initialising training dataset: {dataset_type}")
    dataset = DatasetRegistry.get(dataset_type)(**config["dataset"], mode=mode, datasource=datasource)

    # Log dataset summary
    logging.info(f"Dataset Summary: # of samples = {len(dataset)}")

    return dataset


def _setup_model(config: Dict, dataset: Dataset) -> Model:
    """
    Initialises or loads the model and its descriptors.
    """
    model_config = config["model"]
    model_type = model_config["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f"Loading pretrained model: {model_type}")
        model_params = model_config.get("params", {})
        model = load_pretrained_model(model_type, **model_params)
    else:
        logging.info(f"Initialising model: {model_type}")
        model_params = model_config.get("params", {})

        # Add additional model parameters
        model_params["in_size"] = dataset.x_size
        model_params["out_size"] = dataset.y_size
        model = ModelRegistry.get(model_type)(**model_params)

        # Intialise model weights
        weights_params = model_config.get("weights_params", {})
        weights_type = model_config.get("weights", {})
        kernel = weights_type.get("kernel", "default")
        bias = weights_type.get("bias", "zeros")
        seed = weights_type.get("seed", None)

        logging.info(f"Initialising model weights: {kernel}")
        model.init_model_weights(kernel, bias, seed, **weights_params)

    return model


def _setup_scheme(config: Dict, args, model: Model, dataset: Dataset) -> Learn:
    """
    Initialise training scheme (standard, k-fold, ensemble, etc.).
    """
    model_type = config["model"].get("type")

    logging.info("Initialising training scheme")
    # Pack parameters
    kwargs = {
        "model_config": config.get("model"),
        "hyper_params": config.get("hyperparams", {}),
        "earlystop_params": config.get("earlystop_params", {}),
        "kfold_params": config.get("kfold_params", {}),
        "bootstrap_params": config.get("bootstrap_params", {}),
        "ensemble_params": config.get("ensemble_params", {}),
        "lr_scheduler": config.get("lr_scheduler", False),
        "scheduler_params": config.get("scheduler_params", {}),
        "mlflow": args.mlflow,
        "tensorboard": args.tensorboard,
    }

    scheme = SchemeRegistry.get_learn(model_type)(model, dataset, **kwargs)
    return scheme


###############################################################################
############################## TRAINING STARTER ###############################
###############################################################################


def _run_training(config: Dict, scheme: Learn) -> Tuple[List, str, float]:
    """
    Train using the selected training scheme.
    """
    model_list = []
    start_time = time.time()

    if config["bootstrap"]:
        logging.info("Training model using bootstrap resampling...")
        scheme_type = "bootstrap"
        model_list = scheme.train_bootstrap()
    elif config["ensemble"]:
        logging.info("Training model using ensemble learning...")
        scheme_type = "ensemble"
        model_list = scheme.train_ensemble()
    elif config["kfold"]:
        logging.info("Training model using kfold cross-validation...")
        scheme_type = "kfold"
        model_list.append(scheme.train_kfold())
    else:
        logging.info("Training model using standard training procedure...")
        scheme_type = "std"
        model_list.append(scheme.train_std())

    train_time = time.time() - start_time

    # Move to CPU
    for model in model_list:
        model.to(torch.device("cpu"))

    return model_list, scheme_type, train_time


###############################################################################
############################### SUMMARY LOGGING ###############################
###############################################################################


def _summary_model(model: Model, dataset: Dataset) -> None:
    logging.info("--- Model Summary ---")

    if model.aegan_flag:
        dummy_x = torch.randn(1, dataset.x_size)
        dummy_y = torch.randn(1, dataset.y_size)
        input_data = (dummy_x, dummy_y)
    elif model.batch_flag:
        input_data = None
    else:
        dummy_x = torch.randn(1, dataset.x_size)
        input_data = dummy_x

    summary(model, input_data=input_data)
