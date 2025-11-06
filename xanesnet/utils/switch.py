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

import random
import warnings
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler

from torch import optim
from torch import nn
from typing import Dict

from xanesnet.utils.loss import (
    EMDLoss,
    CosineSimilarityLoss,
    WCCLoss,
    HybridLoss,
    SpectralLossPlus,
)

# Suppress non-significant warning for shap and WCCLoss function
warnings.filterwarnings("ignore")


class ActivationSwitch:
    """
    A factory class to get activation function instances from their names.
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "prelu": nn.PReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }

    def get(self, activation_name: str, **kwargs) -> nn.Module:
        activation_name_lower = activation_name.lower()
        if activation_name_lower not in self.ACTIVATIONS:
            raise TypeError(f"Invalided activation function name '{activation_name}'.")

        activation_class = self.ACTIVATIONS[activation_name_lower]
        return activation_class(**kwargs)


class LossSwitch:
    """
    A factory class to get loss function instances from their names.
    """

    LOSS = {
        "mse": nn.MSELoss,
        "bce": nn.BCEWithLogitsLoss,
        "emd": EMDLoss,
        "cosine": CosineSimilarityLoss,
        "l1": nn.L1Loss,
        "wcc": WCCLoss,
        "hybrid": HybridLoss,
        "specplus": SpectralLossPlus,
    }

    def get(self, loss_name: str, **kwargs) -> nn.Module:
        if loss_name.lower() not in self.LOSS:
            raise TypeError(f"Invalided loss function name '{loss_name}'.")

        loss_class = self.LOSS[loss_name.lower()]

        return loss_class(**kwargs)


class LossRegSwitch:
    """
    Calculates L1 or L2 regularization loss for a model's parameters.
    """

    def loss(self, model, loss_reg_type, device) -> torch.Tensor:
        # Default if loss_reg_type is None or not found
        if not loss_reg_type:
            fn = self._loss_none
        else:
            fn = getattr(self, f"_loss_{loss_reg_type.lower()}", self._loss_none)

        return fn(model, device)

    def _loss_none(self, model, device) -> torch.Tensor:
        # no regularization

        return torch.tensor(0.0, device=device)

    def _loss_l1(self, model, device) -> torch.Tensor:
        # L1 regularization loss (sum of absolute values).
        all_params = torch.cat([p.view(-1) for p in model.parameters()])
        return torch.norm(all_params, p=1)

    def _loss_l2(self, model, device) -> torch.Tensor:
        # L2 regularization loss (sum of squared values).
        all_params = torch.cat([p.view(-1) for p in model.parameters()])
        return torch.norm(all_params, p=2)


class BiasInitSwitch:
    """
    A factory class to get bias initializer functions from their names.
    """

    BIAS = {
        "zeros": nn.init.zeros_,
        "ones": nn.init.ones_,
    }

    def get(self, bias_name: str):
        if bias_name.lower() not in self.BIAS:
            raise TypeError(f"Invalided bias function name '{bias_name}'.")

        return self.BIAS[bias_name.lower()]


class KernelInitSwitch:
    """
    A factory class to get kernel/weight initializer functions from their names.
    """

    KERNEL = {
        "uniform": nn.init.uniform_,
        "normal": nn.init.normal_,
        "xavier_uniform": nn.init.xavier_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "kaiming_normal": nn.init.kaiming_normal_,
    }

    def get(self, kernel_name: str):
        if kernel_name.lower() not in self.KERNEL:
            raise TypeError(f"Invalided kernel function name '{kernel_name}'.")

        return self.KERNEL[kernel_name.lower()]


class LRSchedulerSwitch:
    """
    A factory class to get LRScheduler instance from their names.
    """

    SCHEDULERS = {
        "step": lr_scheduler.StepLR,
        "multistep": lr_scheduler.MultiStepLR,
        "exponential": lr_scheduler.ExponentialLR,
        "linear": lr_scheduler.LinearLR,
        "constant": lr_scheduler.ConstantLR,
    }

    def __init__(self, optimizer, scheduler_type, params=None):
        scheduler_type = scheduler_type.lower()
        params = params or {}

        if scheduler_type not in self.SCHEDULERS:
            raise TypeError(f"Invalided lr scheduler function name '{scheduler_type}'.")

        self.scheduler = self.SCHEDULERS[scheduler_type](optimizer, **params)

    def step(self):
        self.scheduler.step()


class OptimSwitch:
    """
    A factory class to get PyTorch optimizer classes from their names.
    """

    OPTIMIZER = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        "adamw": optim.AdamW,
        "adagrad": optim.Adagrad,
    }

    def get(self, optimizer_name: str):
        if optimizer_name.lower() not in self.OPTIMIZER:
            raise TypeError(f"Invalided lr optimizer name '{optimizer_name}'.")

        return self.OPTIMIZER[optimizer_name.lower()]
