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
import random
import sys
from typing import Tuple, Dict, List

import torch
import time

from datetime import timedelta
from pathlib import Path
from torchinfo import summary

from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.models.base_model import Model
from xanesnet.models.pre_trained import PretrainedModels
from xanesnet.scheme import Learn
from xanesnet.utils.mode import get_mode, Mode
from xanesnet.utils.io import (
    save_models,
    load_pretrained_descriptors,
    load_pretrained_model,
)
from xanesnet.creator import (
    create_learn_scheme,
    create_descriptors,
    create_model,
    create_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        # logging.FileHandler("train.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)


def train(config, args):
    """
    Main training entry
    """
    logging.info(f">> Training mode: {args.mode}")
    mode = get_mode(args.mode)

    # Initialise feature descriptor(s)
    descriptor_list = _setup_descriptors(config)

    # Precess training dataset
    dataset = _setup_dataset(config, mode, descriptor_list)

    # Setup model
    model = _setup_model(config, dataset)

    # Setup training scheme
    scheme = _setup_scheme(config, args, model, dataset)

    # Train the model
    model_list, scheme_type, train_time = _train_model(config, scheme)

    # Display model summary and training duration
    _summary_model(model_list[0], dataset)
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")

    # Save model(s) and metadata to disk if requested
    if args.save:
        metadata = {
            "mode": args.mode,
            "dataset": dataset.config,
            "model": model.config,
            "descriptors": [desc.config for desc in descriptor_list],
            "scheme": scheme_type,
        }

        save_models(Path("models"), model_list, metadata)


def _setup_descriptors(config: Dict) -> List:
    """Initialise or load descriptors depending on the model type."""
    model_type = config["model"]["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f">> Loading pretrained model descriptors: {model_type}")
        descriptor_list = load_pretrained_descriptors(model_type)
    else:
        descriptor_config = config["descriptors"]
        descriptor_types = ", ".join(d["type"] for d in descriptor_config)
        logging.info(f">> Initialising descriptors: {descriptor_types}")
        descriptor_list = create_descriptors(config=descriptor_config)

    return descriptor_list


def _setup_dataset(config: Dict, mode: Mode, descriptor_list: List) -> BaseDataset:
    """Process the dataset using input configuration or load an existing one from disk"""
    dataset_type = config["dataset"]["type"]

    logging.info(f">> Initialising training dataset: {dataset_type}")
    # Pack parameters
    kwargs = {
        "root": config["dataset"]["root_path"],
        "xyz_path": config["dataset"]["xyz_path"],
        "xanes_path": config["dataset"]["xanes_path"],
        "mode": mode,
        "descriptors": descriptor_list,
        **config["dataset"].get("params", {}),
    }

    dataset = create_dataset(dataset_type, **kwargs)

    # Log dataset summary
    logging.info(
        f">> Dataset Summary: # of samples = {len(dataset)}, feature(X) size = {dataset.x_size}, label(y) size = {dataset.y_size}"
    )

    return dataset


def _setup_model(config: Dict, dataset: BaseDataset) -> Model:
    """Initialises or loads the model and its descriptors."""
    model_config = config["model"]
    model_type = model_config["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f">> Loading pretrained model: {model_type}")
        model_params = model_config.get("params", {})
        model = load_pretrained_model(model_type, **model_params)
    else:
        logging.info(f">> Initialising model: {model_type}")
        model_params = model_config.get("params", {})

        # Add additional model parameters
        model_params["in_size"] = dataset.x_size
        model_params["out_size"] = dataset.y_size
        model = create_model(model_type, **model_params)

        # Intialise model weights
        weights_params = model_config.get("weights_params", {})
        weights_type = model_config.get("weights", {})
        kernel = weights_type.get("kernel", "default")
        bias = weights_type.get("bias", "zeros")
        seed = weights_type.get("seed", None)

        logging.info(f">> Initialising model weights: {kernel}")
        model.init_model_weights(kernel, bias, seed, **weights_params)

    return model


def _setup_scheme(config: Dict, args, model: Model, dataset: BaseDataset) -> Learn:
    """Initialise training scheme (standard, k-fold, ensemble, etc.)."""
    model_type = config["model"].get("type")

    logging.info(">> Initialising training scheme")
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

    scheme = create_learn_scheme(model_type, model, dataset, **kwargs)
    return scheme


def _train_model(config: Dict, scheme: Learn) -> Tuple[List, str, float]:
    """
    Train model using the selected training scheme.
    """
    model_list = []
    start_time = time.time()

    if config["bootstrap"]:
        logging.info(">> Training model using bootstrap resampling...\n")
        scheme_type = "bootstrap"
        model_list = scheme.train_bootstrap()
    elif config["ensemble"]:
        logging.info(">> Training model using ensemble learning...\n")
        scheme_type = "ensemble"
        model_list = scheme.train_ensemble()
    elif config["kfold"]:
        logging.info(">> Training model using kfold cross-validation...\n")
        scheme_type = "kfold"
        model_list.append(scheme.train_kfold())
    else:
        logging.info(">> Training model using standard training procedure...\n")
        scheme_type = "std"
        model_list.append(scheme.train_std())

    train_time = time.time() - start_time

    return model_list, scheme_type, train_time


def _summary_model(model: Model, dataset: BaseDataset) -> None:
    logging.info("\n--- Model Summary ---")

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
