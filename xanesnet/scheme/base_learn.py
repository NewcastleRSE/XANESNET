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

import copy
import logging
import random
from dataclasses import dataclass
from typing import Dict, List

import mlflow
import torch
import time
import pickle
import torch_geometric
import numpy as np

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from xanesnet.models.base_model import Model
from xanesnet.utils.switch import (
    OptimSwitch,
    LossSwitch,
    LRSchedulerSwitch,
    LossRegSwitch,
    KernelInitSwitch,
    BiasInitSwitch,
)

# from xanesnet.param_optuna import ParamOptuna
# from xanesnet.param_freeze import Freeze


@dataclass
class EarlyStopState:
    best_loss: float = float("inf")
    no_improve: int = 0
    stop: bool = False


class Learn(ABC):
    """Abstract base class defining the training interface for XANESNET models."""

    def __init__(self, model, dataset, **kwargs):
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.recon_flag = 0  # 1 for AELearn or AEGANLearn
        self.earlystop_flag = 0

        # ---Unpack kwargs
        model_config = kwargs.get("model_config")
        hyper_params = kwargs.get("hyper_params")
        kfold_params = kwargs.get("kfold_params")
        bootstrap_params = kwargs.get("bootstrap_params")
        ensemble_params = kwargs.get("ensemble_params")
        scheduler_params = kwargs.get("scheduler_params")

        # --- Model parameter ---
        self.model_type = model_config.get("type")
        self.model_params = model_config.get("params", {})
        self.weights_type = model_config.get("weights", {})
        self.weights_params = model_config.get("weights_params", {})

        # --- Hyperparameter ---
        self.hyper_params = hyper_params
        self.batch_size = hyper_params.get("batch_size", 32)
        self.epochs = hyper_params.get("epochs", 100)
        self.lr = hyper_params.get("lr", 0.001)
        self.optimizer = hyper_params.get("optimizer", "adam")
        self.model_eval = hyper_params.get("model_eval", False)
        self.loss = hyper_params.get("loss", "mse")
        self.loss_reg = hyper_params.get("loss_reg", "None")
        self.loss_lambda = hyper_params.get("loss_lambda", 0.0001)
        self.n_earlystop = hyper_params.get("n_earlystop", self.epochs)
        self.earlystop_flag = 1 if self.n_earlystop < self.epochs else 0
        self.seed = hyper_params.get("seed", random.randrange(1000))
        self.loss_params = hyper_params.get("loss_params", {})

        # --- K-Fold cross-validation parameters ---
        self.n_splits = kfold_params.get("n_splits", 3)
        self.n_repeats = kfold_params.get("n_repeats", 1)
        self.seed_kfold = kfold_params.get("seed", random.randrange(1000))

        # --- Bootstrap learning parameters ---
        self.n_boot = bootstrap_params.get("n_boot", 3)
        self.n_size = bootstrap_params.get("n_size", 1.0)
        self.weight_seed_boot = bootstrap_params.get(
            "weight_seed", random.sample(range(1000), 3)
        )

        # --- Ensemble learning parameters ---
        self.n_ens = ensemble_params.get("n_ens", 3)
        self.weight_seed_ens = ensemble_params.get(
            "weight_seed", random.sample(range(1000), 3)
        )

        # --- Learning rate scheduler ---
        self.lr_scheduler = kwargs.get("lr_scheduler")
        self.scheduler_type = scheduler_params.get("type")
        self.scheduler_params = {
            k: v for k, v in scheduler_params.items() if k != "type"
        }

        # --- logging ---
        self.mlflow_flag = kwargs.get("mlflow")
        self.tb_flag = kwargs.get("tensorboard")
        self.writer = None

        # Initialize TensorBoard writer
        if self.tb_flag:
            layout = self.tensorboard_layout()
            self.writer = self.setup_writer(layout)

        # Initialize MLflow experiment
        if self.mlflow_flag:
            self.setup_mlflow()

    @abstractmethod
    def tensorboard_layout(self) -> Dict:
        """
        Defines the TensorBoard layout.

        Returns
        -------
        Dict
            A dictionary describing the TensorBoard layout configuration.
        """
        pass

    @abstractmethod
    def train(self, model, dataset) -> float:
        """
        Core training loop.

        Parameters
        ----------
        model : Model
            The untrained model
        dataset : BaseDataset
            XANES dataset

        Returns
        -------
        float
            The final score - validation loss from the last epoch
        """
        pass

    @abstractmethod
    def train_std(self) -> Model:
        """
        Trains model using standard single-run training.

        Returns
        -------
        Model
            The trained model object.
        """
        pass

    @abstractmethod
    def train_kfold(self) -> Model:
        """
        Trains model using k-fold cross-validation.


        Returns
        -------
        Model
            The (best) trained model object.
        """
        pass

    def train_bootstrap(self) -> List[Model]:
        """

        Trains models using bootstrap resampling.

        Returns
        -------
        List[Model]
            A list of trained models.
        """

        model_list = []
        n_samples = len(self.dataset)
        sample_size = int(n_samples * self.n_size)

        for i in range(self.n_boot):
            rng = np.random.default_rng(self.weight_seed_boot[i])

            # Generate all random indices at once
            bootstrap_indices = rng.choice(n_samples, size=sample_size, replace=True)
            dataset_boot = self.dataset[bootstrap_indices]

            # Deep copy model
            model = copy.deepcopy(self.model)

            # Re-initialise model weight using ensemble seeds
            kernel = self.weights_type.get("kernel", "default")
            bias = self.weights_type.get("bias", "zeros")
            seed = self.weight_seed_boot[i]
            model.init_model_weights(kernel, bias, seed, **self.weights_params)

            self.train(model, dataset_boot)

            model_list.append(model)

        return model_list

    def train_ensemble(self) -> List[Model]:
        """
        Trains an ensemble of models with different random weight initialisations

        Returns
        -------
        List[Model]
            A list of trained models.
        """
        model_list = []

        for i in range(self.n_ens):
            # Deep copy model
            model = copy.deepcopy(self.model)

            # Re-initialise model weight using ensemble seeds
            kernel = self.weights_type.get("kernel", "default")
            bias = self.weights_type.get("bias", "zeros")
            seed = self.weight_seed_ens[i]
            model.init_model_weights(kernel, bias, seed, **self.weights_params)

            self.train(model, self.dataset)
            model_list.append(model)

        return model_list

    def setup_components(self, model):
        """Initialises the optimizer, loss function, regularizer, and scheduler."""
        # --- Initialise Optimizer ---
        optim_fn = OptimSwitch().get(self.optimizer)
        optimizer = optim_fn(model.parameters(), self.lr)

        # --- Initialise loss functions ---
        criterion = LossSwitch().get(self.loss, **self.loss_params)

        # --- Regularizer ---
        regularizer = LossRegSwitch()

        # --- LR schedulers (optional) ---
        scheduler = None
        if self.lr_scheduler:
            scheduler = LRSchedulerSwitch(
                optimizer,
                self.scheduler_type,
                self.scheduler_params,
            )

        return optimizer, criterion, regularizer, scheduler

    def setup_dataloaders(self, dataset):
        """Splits the dataset and creates DataLoaders."""
        indices = dataset.indices

        if self.model_eval:
            # Data split: train/valid/test
            train_ratio, valid_ratio, eval_ratio = 0.75, 0.15, 0.10

            # First split: train vs (test + eval)
            train_idx, valid_idx = train_test_split(
                indices, test_size=1 - train_ratio, random_state=self.seed
            )

            # Second split: test vs eval
            test_size = eval_ratio / (eval_ratio + valid_ratio)
            valid_idx, eval_idx = train_test_split(
                valid_idx, test_size=test_size, random_state=self.seed
            )
        else:
            train_idx, valid_idx = train_test_split(
                indices, test_size=0.2, random_state=self.seed
            )

            eval_idx = None

        train_loader = self._create_loader(dataset[train_idx], shuffle=True)
        valid_loader = self._create_loader(dataset[valid_idx])
        eval_loader = self._create_loader(dataset[eval_idx]) if eval_idx else None

        return train_loader, valid_loader, eval_loader

    def _create_loader(self, dataset, shuffle: bool = False, drop_last: bool = False):
        """Creates a DataLoader based on model type."""
        if self.model.gnn_flag:
            dataloader_cls = torch_geometric.data.DataLoader
        else:
            dataloader_cls = torch.utils.data.DataLoader

        return dataloader_cls(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last,
        )

    def setup_mlflow(self):
        """Initialises an MLflow experiment."""
        experiment_name = self.model.__class__.__name__
        mlflow.set_experiment(experiment_name)

        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(self.hyper_params)
        mlflow.log_param("n_epoch", self.epochs)

    def setup_writer(self, layout: Dict) -> SummaryWriter:
        """Initializes a TensorBoard SummaryWriter with a given layout."""
        writer = SummaryWriter(f"tensorboard/{int(time.time())}")
        writer.add_custom_scalars(layout)

        return writer

    def _early_stop(self, val_loss, state):
        if val_loss < state.best_loss:
            state.best_loss = val_loss
            state.no_improve = 0
        else:
            state.no_improve += 1
            if state.no_improve >= self.n_earlystop:
                logging.info(f"Early stopping after {self.n_earlystop} epochs.")
                state.stop = True

    def log_mlflow(self, model):
        """Log the model as an artifact of the MLflow run."""
        mlflow.pytorch.log_model(
            model, artifact_path="pytorch-model", pickle_module=pickle
        )

        mlflow.pytorch.load_model(mlflow.get_artifact_uri("pytorch-model"))

    def log_loss(self, name: str, value: float, epoch: int):
        """Log loss to MLflow and/or TensorBoard."""
        if self.mlflow_flag:
            mlflow.log_metric(name, value, step=epoch)

        if self.tb_flag:
            self.writer.add_scalar(name, value, epoch)

    def log_close(self):
        """Finalises and closes TensorBoard and MLflow logging sessions."""
        if self.tb_flag:
            log_dir = self.writer.log_dir  # Get TensorBoard log directory
            self.writer.close()
            print(f"\nTensorBoard logs saved at: file://{Path(log_dir).resolve()}")

        if self.mlflow_flag:
            run_url = mlflow.get_artifact_uri()  # Get the MLflow run URL
            mlflow.end_run()
            print(f"\nMLflow run saved at: {run_url}")
