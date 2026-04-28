# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Singleton TensorBoard logger for XANESNET training runs."""

import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from xanesnet.serialization.config import Config


class _TensorBoardGraphWrapper(torch.nn.Module):
    """Wraps a model so TensorBoard graph tracing receives only tensor inputs.

    Non-tensor inputs are bound once at construction time and injected
    transparently on each ``forward`` call.

    Args:
        model: The model to wrap.
        input_example: A dict mapping forward-argument names to example values.
    """

    def __init__(self, model: torch.nn.Module, input_example: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self._ordered_arg_names: list[str] = []
        self._tensor_arg_names: list[str] = []
        self._static_arg_values: dict[str, Any] = {}

        for name, parameter in inspect.signature(model.forward).parameters.items():
            if name == "self":
                continue

            if parameter.kind not in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                raise TypeError(f"TensorBoard graph logging does not support variadic forward parameter '{name}'.")

            if name not in input_example:
                if parameter.default is inspect.Parameter.empty:
                    raise ValueError(f"Missing model input '{name}' required for TensorBoard graph logging.")
                continue

            value = input_example[name]
            self._ordered_arg_names.append(name)

            if isinstance(value, torch.Tensor):
                self._tensor_arg_names.append(name)
                continue

            if isinstance(value, torch.nn.Module):
                module_name = f"_graph_static_module_{len(self._static_arg_values)}"
                self.add_module(module_name, value)
                value = getattr(self, module_name)

            self._static_arg_values[name] = value

    @property
    def tensor_arg_names(self) -> tuple[str, ...]:
        """Names of the tensor-valued forward arguments, in call order."""
        return tuple(self._tensor_arg_names)

    def forward(self, *tensor_args: torch.Tensor) -> Any:
        """Forward pass with bound non-tensor arguments re-injected automatically.

        Args:
            *tensor_args: Tensor inputs in the order returned by
                ``tensor_arg_names``.

        Returns:
            The output of the wrapped model.
        """
        if len(tensor_args) != len(self._tensor_arg_names):
            raise ValueError(
                f"Expected {len(self._tensor_arg_names)} tensor inputs for TensorBoard graph logging, "
                f"got {len(tensor_args)}."
            )

        tensor_arg_values = dict(zip(self._tensor_arg_names, tensor_args, strict=True))
        ordered_args = [
            tensor_arg_values[name] if name in tensor_arg_values else self._static_arg_values[name]
            for name in self._ordered_arg_names
        ]
        return self.model(*ordered_args)


class TensorBoardLogger:
    """Singleton TensorBoard logger for XANESNET training runs.

    Use ``new_run`` to initialise logging for each new training run.  All
    ``log_*`` methods silently no-op when the logger has not been initialised
    or when TensorBoard has been disabled.

    Note:
        Instantiate via ``tb_logger = TensorBoardLogger()`` and import the
        module-level singleton rather than creating fresh instances.
    """

    _instance: "TensorBoardLogger | None" = None

    _writer: SummaryWriter | None
    _config: Config | None
    _enabled: bool

    def __new__(cls) -> "TensorBoardLogger":
        if cls._instance is None:
            cls._instance = super(TensorBoardLogger, cls).__new__(cls)
            cls._instance._writer = None
            cls._instance._config = None
            cls._instance._enabled = False
        return cls._instance

    @property
    def enabled(self) -> bool:
        """Whether TensorBoard logging is currently active for this process."""
        return self._enabled

    def set_config(self, config: Config) -> None:
        """Attach the run configuration used for later hyperparameter logging.

        Args:
            config: Validated configuration for the upcoming run.
        """
        self._config = config

    def new_run(self, save_dir: str | Path) -> None:
        """Initialise a new TensorBoard run.

        Closes any previously open writer, creates a new ``SummaryWriter``
        pointing at ``save_dir``, and logs a hyperparameter table if a config
        has been set via ``set_config``.

        Args:
            save_dir: Directory in which TensorBoard event files will be
                written.
        """
        if self._writer is not None:
            self._writer.close()

        self._writer = SummaryWriter(log_dir=str(save_dir))
        self._enabled = True

        # Log hyperparameters as text (nested-dict safe, unlike add_hparams)
        if self._config is not None:
            flat = self._flatten_dict(self._config.as_dict())
            # Write a markdown table of hyperparameters
            rows = [f"| `{k}` | `{v}` |" for k, v in sorted(flat.items())]
            md = "| Hyperparameter | Value |\n|---|---|\n" + "\n".join(rows)
            self._writer.add_text("hyperparameters", md, global_step=0)

        logging.debug(f"Initialized new TensorBoard SummaryWriter with log_dir: {save_dir}")

    def close(self) -> None:
        """Flush pending events and close the underlying ``SummaryWriter``."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
        self._enabled = False

    def _check_initialized(self) -> None:
        """Raise ``RuntimeError`` if ``new_run`` has not been called yet."""
        if self._writer is None:
            raise RuntimeError("TensorBoardLogger not initialized. Call new_run() first.")

    @property
    def writer(self) -> SummaryWriter:
        """Return the active ``SummaryWriter``.

        Returns:
            The current ``SummaryWriter`` instance.

        Raises:
            RuntimeError: If ``new_run()`` has not been called.
        """
        self._check_initialized()
        assert self._writer is not None  # for type narrowing
        return self._writer

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_regularization: float,
        train_total: float,
        valid_loss: float | None = None,
        valid_regularization: float | None = None,
        valid_total: float | None = None,
    ) -> None:
        """Log per-epoch training and optional validation metrics.

        Args:
            epoch: Current epoch index (used as the global step).
            train_loss: Training data loss.
            train_regularization: Training regularization term.
            train_total: Total training loss (loss + regularization).
            valid_loss: Validation data loss (``None`` to skip).
            valid_regularization: Validation regularization term (``None`` to skip).
            valid_total: Total validation loss (``None`` to skip).
        """
        if not self._enabled:
            return

        w = self.writer

        # Training scalars
        w.add_scalar("loss/train", train_loss, epoch)
        w.add_scalar("regularization/train", train_regularization, epoch)
        w.add_scalar("total/train", train_total, epoch)

        # Validation scalars
        if valid_total is not None:
            w.add_scalar("loss/valid", valid_loss, epoch)
            w.add_scalar("regularization/valid", valid_regularization, epoch)
            w.add_scalar("total/valid", valid_total, epoch)

    def log_learning_rate(self, epoch: int, lr: float) -> None:
        """Log the current learning rate at the given epoch.

        Args:
            epoch: Current epoch index.
            lr: Learning rate value.
        """
        if not self._enabled:
            return

        self.writer.add_scalar("other/learning_rate", lr, epoch)

    def log_model_weights(self, epoch: int, model: torch.nn.Module) -> None:
        """Log histograms of model parameter values and gradients.

        Args:
            epoch: Current epoch index (used as the global step).
            model: The model whose parameters are logged.
        """
        if not self._enabled:
            return

        w = self.writer
        for name, param in model.named_parameters():
            tag = name.replace(".", "/")
            w.add_histogram(f"weights/{tag}", param.data, epoch)
            if param.grad is not None:
                w.add_histogram(f"gradients/{tag}", param.grad.data, epoch)

    def log_model_graph(self, model: torch.nn.Module, input_example: dict[str, Any]) -> None:
        """Log the model computation graph (call once at the start of training).

        Inputs are ordered according to the model's ``forward`` signature.
        Non-tensor inputs are bound automatically using
        ``_TensorBoardGraphWrapper``. Failures during tracing are caught and
        logged as warnings rather than propagated.

        Args:
            model: The model to trace.
            input_example: Dict mapping forward-argument names to example
                values.
        """
        if not self._enabled:
            return

        try:
            graph_model = _TensorBoardGraphWrapper(model, input_example)
            graph_inputs = tuple(input_example[name] for name in graph_model.tensor_arg_names)

            if not graph_inputs:
                logging.warning(
                    "Skipping TensorBoard graph logging because the model has no tensor inputs to trace."
                )
                return

            non_tensor_input_names = [
                name for name in graph_model._ordered_arg_names if name not in graph_model.tensor_arg_names
            ]
            if non_tensor_input_names:
                logging.info(
                    "Binding non-tensor model inputs for TensorBoard graph logging: %s",
                    ", ".join(non_tensor_input_names),
                )

            self.writer.add_graph(graph_model, graph_inputs)
            logging.info("Logged model graph to TensorBoard.")
        except Exception as exc:
            logging.warning("Skipping TensorBoard graph logging because tracing failed: %s", exc)

    # PRIMITIVE LOGGING FUNCTIONS

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value.

        Args:
            tag: TensorBoard tag (used as the series name).
            value: Scalar value to record.
            step: Global step / epoch index.
        """
        if not self._enabled:
            return

        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram of tensor values.

        Args:
            tag: TensorBoard tag.
            values: Tensor whose distribution is recorded.
            step: Global step / epoch index.
        """
        if not self._enabled:
            return

        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text string.

        Args:
            tag: TensorBoard tag.
            text: The text content to record.
            step: Global step / epoch index.
        """
        if not self._enabled:
            return

        self.writer.add_text(tag, text, step)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure as an image.

        Args:
            tag: TensorBoard tag.
            figure: A ``matplotlib.figure.Figure`` instance.
            step: Global step / epoch index.
        """
        if not self._enabled:
            return

        self.writer.add_figure(tag, figure, step)

    def log_final_metrics(self, metric_dict: dict[str, float]) -> None:
        """Log final run-level metrics alongside hyperparameters.

        Uses ``add_hparams`` so that hyperparameters and final metrics appear
        in the TensorBoard HPARAMS tab.  Call once at the end of training.

        Args:
            metric_dict: Mapping from metric name to final scalar value.
        """
        if not self._enabled:
            return

        if self._config is not None:
            flat_hparams = self._flatten_dict(self._config.as_dict())
            # add_hparams only accepts str/bool/int/float/Tensor values
            hparams = {k: v for k, v in flat_hparams.items() if isinstance(v, (str, bool, int, float))}
            # run_name="." prevents add_hparams from creating a sub-directory
            self.writer.add_hparams(hparams, metric_dict, run_name=".")

    def flush(self) -> None:
        """Flush pending events to disk."""
        if self._writer is not None:
            self._writer.flush()

    @staticmethod
    def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        """Flatten a nested dictionary into dot-separated keys.

        Args:
            d: Dictionary to flatten.
            parent_key: Prefix prepended to nested keys.
            sep: Separator inserted between nested key segments.

        Returns:
            Flat dictionary whose keys encode the original nesting.
        """
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(TensorBoardLogger._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                # Store lists as their string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


###############################################################################
################################## SINGLETON ##################################
###############################################################################

tb_logger = TensorBoardLogger()
