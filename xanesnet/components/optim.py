# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Optimizers and registry for XANESNET training."""

# TODO: Maybe there is a better way to implement this

from abc import ABC, abstractmethod
from typing import Any, cast

import torch.optim as optim

from xanesnet.serialization.config import Config
from xanesnet.utils.registry import Registry


class Optimizer(ABC):
    """Mixin adding XANESNET signatures to native PyTorch optimizers."""

    optimizer_type: str
    _signature: Config

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return the native optimizer state for checkpointing."""
        ...

    @property
    def signature(self) -> Config:
        """Return the optimizer type required to reconstruct this component.

        Returns:
            Minimal configuration containing the optimizer registry key.
        """
        return self._signature

    @property
    def torch_optimizer(self) -> optim.Optimizer:
        """Return this optimizer for framework integrations.

        Returns:
            This native PyTorch optimizer instance.
        """
        return cast(optim.Optimizer, self)

    def _set_optimizer_type(self, optimizer_type: str) -> None:
        """Store the optimizer type and its reconstruction signature.

        Args:
            optimizer_type: Registry key identifying the optimizer type.
        """
        self.optimizer_type = optimizer_type.lower()
        self._signature = Config({"optimizer": self.optimizer_type})


OptimizerRegistry: Registry[type[Optimizer]] = Registry("Optimizer", normalize_key=str.lower)


@OptimizerRegistry.register("adam")
class AdamOptimizer(optim.Adam, Optimizer):
    """Adam optimizer registered with XANESNET.

    Args:
        optimizer_type: Registry key identifying this optimizer type; must be ``"adam"``.
        params: Parameters or parameter groups to optimize.
        lr: Learning rate.
        **kwargs: Additional keyword arguments forwarded to ``torch.optim.Adam``.
    """

    def __init__(self, optimizer_type: str, params: Any, lr: float, **kwargs: Any) -> None:
        """Initialize ``AdamOptimizer``."""
        super().__init__(params=params, lr=lr, **kwargs)
        self._set_optimizer_type(optimizer_type)


@OptimizerRegistry.register("sgd")
class SGDOptimizer(optim.SGD, Optimizer):
    """Stochastic gradient descent optimizer registered with XANESNET.

    Args:
        optimizer_type: Registry key identifying this optimizer type; must be ``"sgd"``.
        params: Parameters or parameter groups to optimize.
        lr: Learning rate.
        **kwargs: Additional keyword arguments forwarded to ``torch.optim.SGD``.
    """

    def __init__(self, optimizer_type: str, params: Any, lr: float, **kwargs: Any) -> None:
        """Initialize ``SGDOptimizer``."""
        super().__init__(params=params, lr=lr, **kwargs)
        self._set_optimizer_type(optimizer_type)


@OptimizerRegistry.register("rmsprop")
class RMSpropOptimizer(optim.RMSprop, Optimizer):
    """RMSprop optimizer registered with XANESNET.

    Args:
        optimizer_type: Registry key identifying this optimizer type; must be ``"rmsprop"``.
        params: Parameters or parameter groups to optimize.
        lr: Learning rate.
        **kwargs: Additional keyword arguments forwarded to ``torch.optim.RMSprop``.
    """

    def __init__(self, optimizer_type: str, params: Any, lr: float, **kwargs: Any) -> None:
        """Initialize ``RMSpropOptimizer``."""
        super().__init__(params=params, lr=lr, **kwargs)
        self._set_optimizer_type(optimizer_type)


@OptimizerRegistry.register("adamw")
class AdamWOptimizer(optim.AdamW, Optimizer):
    """AdamW optimizer registered with XANESNET.

    Args:
        optimizer_type: Registry key identifying this optimizer type; must be ``"adamw"``.
        params: Parameters or parameter groups to optimize.
        lr: Learning rate.
        **kwargs: Additional keyword arguments forwarded to ``torch.optim.AdamW``.
    """

    def __init__(self, optimizer_type: str, params: Any, lr: float, **kwargs: Any) -> None:
        """Initialize ``AdamWOptimizer``."""
        super().__init__(params=params, lr=lr, **kwargs)
        self._set_optimizer_type(optimizer_type)


@OptimizerRegistry.register("adagrad")
class AdagradOptimizer(optim.Adagrad, Optimizer):
    """Adagrad optimizer registered with XANESNET.

    Args:
        optimizer_type: Registry key identifying this optimizer type; must be ``"adagrad"``.
        params: Parameters or parameter groups to optimize.
        lr: Learning rate.
        **kwargs: Additional keyword arguments forwarded to ``torch.optim.Adagrad``.
    """

    def __init__(self, optimizer_type: str, params: Any, lr: float, **kwargs: Any) -> None:
        """Initialize ``AdagradOptimizer``."""
        super().__init__(params=params, lr=lr, **kwargs)
        self._set_optimizer_type(optimizer_type)
