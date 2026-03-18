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


def train(config: Dict, args):
    """
    Main training entry
    """
    logging.info(f">> Training mode: {args.mode}")
    mode = get_mode(args.mode)

    # Initialise feature descriptor(s)
    descriptors = setup_descriptors(config)

    # Precess training dataset
    dataset = setup_dataset(config, mode, descriptors)

    # Setup model
    model = setup_model(config, dataset)

    # Setup training scheme
    scheme = setup_scheme(config, args, model, dataset)

    # Train the model
    models, scheme_type, train_time = train_model(config, scheme)

    # Display model summary and training duration
    summarise_model(models[0], dataset)
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")

    # Save model(s) and metadata to disk if requested
    if args.save:
        save_training_outputs(
            models=models,
            scheme_type=scheme_type,
            dataset=dataset,
            model=model,
            descriptors=descriptors,
            mode=args.mode,
        )


def setup_descriptors(config: Dict) -> List:
    """Initialise or load descriptors depending on the model type."""
    model_type = config["model"]["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f">> Loading pretrained model descriptors: {model_type}")
        return load_pretrained_descriptors(model_type)

    descriptor_cfg = config.get("descriptors", None)

    if descriptor_cfg is None or descriptor_cfg == "none":
        logging.info(">> No descriptors used (descriptor: none)")
        return []  # IMPORTANT: return empty list, not None

    if isinstance(descriptor_cfg, list):
        if len(descriptor_cfg) == 0 or str(descriptor_cfg[0].get("type", "")).lower() == "none":
            logging.info(">> No descriptors used (descriptor: none)")
            return []

    descriptor_types = ", ".join(d["type"] for d in descriptor_cfg)
    logging.info(f">> Initialising descriptors: {descriptor_types}")

def setup_dataset(config: Dict, mode: Mode, descriptors: List) -> BaseDataset:
    """Create and preprocess dataset."""
    dataset_cfg = config["dataset"]
    dataset_type = dataset_cfg["type"]

    logging.info(f">> Initialising dataset: {dataset_type}")

    dataset = create_dataset(
        dataset_type,
        root=dataset_cfg["root_path"],
        xyz_path=dataset_cfg["xyz_path"],
        xanes_path=dataset_cfg["xanes_path"],
        mode=mode,
        descriptors=descriptors,
        **dataset_cfg.get("params", {}),
    )

    # Log dataset summary
    logging.info(
        ">> Dataset Summary: # of samples=%d, X=%s, y=%s",
        len(dataset),
        dataset.in_features,
        dataset.out_features,
    )

    return dataset


def setup_model(config: Dict, dataset: BaseDataset) -> Model:
    """Initialise or load model."""
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    model_params = model_cfg.get("params", {})

    if hasattr(PretrainedModels, model_type):
        logging.info(">> Loading pretrained model: %s", model_type)
        return load_pretrained_model(model_type, **model_params)

    logging.info(">> Initialising model: %s", model_type)
    model_params = {
        **model_params,
        "in_features": dataset.in_features,
        "out_features": dataset.out_features,
    }

    model = create_model(model_type, **model_params)
    initialise_weights(model, model_cfg)

    return model


def initialise_weights(model: Model, model_cfg: Dict) -> None:
    """Initialise model weights."""
    weights_cfg = model_cfg.get("weights", {})
    weights_params = model_cfg.get("weights_params", {})

    kernel = weights_cfg.get("kernel", "default")
    bias = weights_cfg.get("bias", "zeros")
    seed = weights_cfg.get("seed")

    logging.info(">> Initialising weights: kernel=%s", kernel)
    model.init_model_weights(kernel, bias, seed, **weights_params)


def setup_scheme(config: Dict, args, model: Model, dataset: BaseDataset) -> Learn:
    """Create training scheme."""
    logging.info(">> Initialising training scheme")
    return create_learn_scheme(
        config["model"]["type"],
        model,
        dataset,
        model_config=config.get("model"),
        hyper_params=config.get("hyperparams", {}),
        earlystop_params=config.get("earlystop_params", {}),
        kfold_params=config.get("kfold_params", {}),
        bootstrap_params=config.get("bootstrap_params", {}),
        ensemble_params=config.get("ensemble_params", {}),
        lr_scheduler=config.get("lr_scheduler", False),
        scheduler_params=config.get("scheduler_params", {}),
        mlflow=args.mlflow,
        tensorboard=args.tensorboard,
    )


def train_model(config: Dict, scheme: Learn) -> Tuple[List, str, float]:
    """
    Train model using the selected training scheme.
    """
    start_time = time.time()

    if config["bootstrap"]:
        logging.info(">> Bootstrap training...\n")
        models = scheme.train_bootstrap()
        scheme_type = "bootstrap"

    elif config["ensemble"]:
        logging.info(">> Ensemble training...\n")
        models = scheme.train_ensemble()
        scheme_type = "ensemble"

    elif config.get("kfold"):
        logging.info(">> K-fold training...\n")
        models = [scheme.train_kfold()]
        scheme_type = "kfold"

    else:
        logging.info(">> Standard training...\n")
        models = [scheme.train_std()]
        scheme_type = "std"

    train_time = time.time() - start_time

    return models, scheme_type, train_time


def summarise_model(model: Model, dataset: BaseDataset) -> None:
    """Print torchinfo summary."""
    logging.info("\n--- Model Summary ---")

    if model.aegan_flag:
        dummy_x = torch.randn(1, dataset.in_features)
        dummy_y = torch.randn(1, dataset.out_features)
        input_data = (dummy_x, dummy_y)
    elif model.batch_flag:
        input_data = None
    else:
        dummy_x = torch.randn(1, dataset.in_features)
        input_data = dummy_x

    summary(model, input_data=input_data)


def save_training_outputs(
    *,
    models: List[Model],
    scheme_type: str,
    dataset: BaseDataset,
    model: Model,
    descriptors: List,
    mode: str,
) -> None:
    """Save models and metadata to disk"""
    metadata = {
        "mode": mode,
        "dataset": dataset.config,
        "model": model.config,
        "descriptors": [d.config for d in descriptors],
        "scheme": scheme_type,
    }

    save_models(
        path=Path("models"),
        models=models,
        metadata=metadata,
        gauss_basis=dataset.gauss_basis,
    )
