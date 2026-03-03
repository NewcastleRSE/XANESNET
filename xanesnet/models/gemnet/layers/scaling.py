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

import json
import logging
from typing import Any

import torch


class ScalingFactor(torch.nn.Module):
    """
    Module that applies a stored scale factor and registers observations
    for automatic fitting.

    When ``scale_file`` is ``None`` scaling is disabled: the factor stays
    at 1.0 and no observations are tracked.
    """

    def __init__(
        self,
        scale_file: str | None,
        name: str,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)

        if scale_file is not None:
            self.autofit: AutoScaleFit | None = AutoScaleFit(self.scale_factor, scale_file, name)
        else:
            self.autofit = None

    def forward(self, x_ref: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Multiply output by scale factor and register observation for fitting.
        y = y * self.scale_factor
        if self.autofit is not None:
            self.autofit.observe(x_ref, y)
        return y


class AutomaticFit:
    """
    Manage a queue of variables that should be automatically fitted.
    Instances are registered in creation order when fitting mode is active
    and the instance is not already fitted (loaded).
    """

    # Class-level state used to manage the active variable and the queue.
    activeVar: "AutomaticFit | None" = None
    queue: list["AutomaticFit"] | None = None
    fitting_mode: bool = False
    all_processed: bool = False

    def __init__(
        self,
        variable: torch.Tensor,
        scale_file: str,
        name: str,
    ) -> None:
        self.variable = variable
        self.scale_file: str = scale_file
        self.name: str = name

        self._fitted: bool = False
        self.load_maybe()

        # When fitting mode is on and this variable wasn't loaded from file,
        # register this instance in the global queue.
        if AutomaticFit.fitting_mode and not self._fitted:
            if AutomaticFit.activeVar is None:
                AutomaticFit.activeVar = self
                AutomaticFit.queue = []  # initialize queue when the first item appears
            else:
                self._add2queue()

    def load_maybe(self) -> None:
        """
        Try to load a previously stored scaling factor from JSON. If a value is found,
        set the variable and mark this instance as fitted. Otherwise, log the initialization.
        """
        value = _read_value_json(self.scale_file, self.name)
        if value is None:
            current_val = float(self.variable.detach().cpu())
            logging.info(f"Initialize variable '{self.name}' to {current_val:.3f}")
        else:
            self._fitted = True
            logging.debug(f"Set scale factor {self.name} : {value}")
            with torch.no_grad():
                self.variable.copy_(torch.tensor(value))

    def _add2queue(self) -> None:
        """
        Add self to the global queue. Raise if a variable with the same name exists.
        """
        logging.debug(f"Add {self.name} to queue.")
        if AutomaticFit.queue is None:
            # This should not happen in normal flow, but guard defensively.
            AutomaticFit.queue = []

        # check that same variable is not added twice
        for var in AutomaticFit.queue:
            if self.name == var.name:
                raise ValueError(f"Variable with the same name ({self.name}) was already added to queue!")
        AutomaticFit.queue.append(self)

    def set_next_active(self) -> None:
        """
        Set the next variable in the queue as active. If the queue is empty,
        mark processing as finished by setting queue to None and clearing activeVar.
        """
        queue = AutomaticFit.queue

        # If queue is None, nothing to do.
        if queue is None:
            return

        if len(queue) == 0:
            logging.debug("Processed all variables.")
            AutomaticFit.queue = None
            AutomaticFit.activeVar = None
            return

        AutomaticFit.activeVar = queue.pop(0)

    @classmethod
    def reset(cls) -> None:
        cls.activeVar = None
        cls.all_processed = False

    @classmethod
    def fitting_completed(cls) -> bool:
        return cls.queue is None

    @classmethod
    def set2fitmode(cls) -> None:
        cls.reset()
        cls.fitting_mode = True


class AutoScaleFit(AutomaticFit):
    """
    Automatically fit scaling factors based on observed variances.
    """

    def __init__(
        self,
        variable: torch.Tensor,
        scale_file: str,
        name: str,
    ) -> None:
        super().__init__(variable, scale_file, name)

        if not self._fitted:
            self._init_stats()

    def _init_stats(self) -> None:
        # variance accumulators are lazily initialized on first observation to match device
        self.variance_in: torch.Tensor | None = None
        self.variance_out: torch.Tensor | None = None
        self.nSamples: int = 0

    def observe(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Observe variances for input x and output y and accumulate them.
        Only accumulates statistics when this instance is the active variable
        and it's not already fitted.
        """
        if self._fitted:
            return

        # only track stats for the currently active variable
        if AutomaticFit.activeVar == self:
            nSamples = int(y.shape[0])
            device = y.device

            # lazily initialize accumulators on the device of the observed tensors
            if self.variance_in is None:
                self.variance_in = torch.tensor(0.0, device=device)
            if self.variance_out is None:
                self.variance_out = torch.tensor(0.0, device=device)

            # mean of variances across batch features, then scaled by number of samples
            # (mirrors original implementation).
            var_in_batch = torch.mean(torch.var(x, dim=0)) * nSamples
            var_out_batch = torch.mean(torch.var(y, dim=0)) * nSamples

            self.variance_in = self.variance_in + var_in_batch
            self.variance_out = self.variance_out + var_out_batch
            self.nSamples += nSamples

    def fit(self) -> None:
        """
        Fit the scaling factor based on the accumulated variances.
        Applies the scaling to `self.variable`, updates the JSON on disk,
        and advances the global queue to the next active variable.
        """
        if AutomaticFit.activeVar == self:
            if (
                self.variance_in is None
                or float(self.variance_in) == 0.0
                or self.variance_out is None
                or float(self.variance_out) == 0.0
            ):
                raise ValueError(
                    f"Did not track the variable {self.name}. Add observe calls to track the variance before and after."
                )

            # compute mean variances across all observed samples
            variance_in_mean = self.variance_in / self.nSamples
            variance_out_mean = self.variance_out / self.nSamples

            ratio = variance_out_mean / variance_in_mean
            value = torch.sqrt(1.0 / ratio)

            # Logging the values (move to CPU for safe float conversion)
            logging.info(
                f"Variable: {self.name}, Var_in: {float(variance_in_mean.cpu()):.3f}, "
                f"Var_out: {float(variance_out_mean.cpu()):.3f}, "
                f"Ratio: {float(ratio.cpu()):.3f} => Scaling factor: {float(value.cpu()):.3f}"
            )

            # Apply scaling to the variable in-place (no grads).
            with torch.no_grad():
                self.variable.mul_(value)

            # Persist the resulting scalar value to JSON
            _update_json(self.scale_file, {self.name: float(self.variable.detach().cpu())})

            # Move to the next variable in the queue (if any)
            self.set_next_active()


###############################################################################
################################### HELPERS ###################################
###############################################################################

# TODO remove 'Any' type hints and replace with more specific types if possible


def _ensure_json_path(path: str) -> None:
    if not path.endswith(".json"):
        raise ValueError(f"Path {path} is not a json-path.")


def _read_json(path: str) -> dict[str, Any]:
    _ensure_json_path(path)

    with open(path, "r", encoding="utf-8") as f:
        content: dict[str, Any] = json.load(f)
    return content


def _write_json(path: str, data: dict[str, Any]) -> None:
    _ensure_json_path(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _update_json(path: str, data: dict[str, Any]) -> None:
    _ensure_json_path(path)

    content = _read_json(path)
    content.update(data)
    _write_json(path, content)


def _read_value_json(path: str, key: str) -> Any | None:
    content = _read_json(path)

    return content.get(key, None)
