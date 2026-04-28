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

from typing import Any

import numpy as np
import torch

from xanesnet.losses import LossRegistry
from xanesnet.serialization.prediction_readers import PredictionSample

from .base import Collector
from .registry import CollectorRegistry


@CollectorRegistry.register("error_metric")
class ErrorMetrics(Collector):
    """
    Compute loss values between predictions and targets using xanesnet.losses.
    """

    def __init__(
        self,
        collector_type: str,
        loss_type: str,
        **loss_kwargs: Any,
    ) -> None:
        super().__init__(collector_type)

        self.loss_type = loss_type

        # Initialize loss function
        loss_class = LossRegistry.get(loss_type)
        self.loss_fn = loss_class(loss_type=loss_type, **loss_kwargs)

    def process(self, sample: PredictionSample) -> dict[str, float]:
        pred = sample["prediction"]
        target = sample["target"]

        # Convert to torch tensors
        if isinstance(pred, np.ndarray):
            pred_torch = torch.from_numpy(pred)
        elif isinstance(pred, torch.Tensor):
            pred_torch = pred
        else:
            pred_torch = torch.tensor(pred, dtype=torch.float32)

        if isinstance(target, np.ndarray):
            target_torch = torch.from_numpy(target)
        elif isinstance(target, torch.Tensor):
            target_torch = target
        else:
            target_torch = torch.tensor(target, dtype=torch.float32)

        # Ensure float32
        pred_torch = pred_torch.float()
        target_torch = target_torch.float()

        # Losses operate on batched spectral tensors with shape (B, N).
        if pred_torch.ndim == 1:
            pred_torch = pred_torch.unsqueeze(0)
        if target_torch.ndim == 1:
            target_torch = target_torch.unsqueeze(0)

        # Compute loss
        loss_value = self.loss_fn(pred_torch, target_torch)

        return {self.loss_type: float(loss_value.item())}
