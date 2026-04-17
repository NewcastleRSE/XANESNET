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

import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from xanesnet.serialization.config import Config


class _TensorBoardGraphWrapper(torch.nn.Module):
    """
    Wrap a model so TensorBoard graph tracing only receives tensor inputs.

    Non-tensor inputs are bound once during construction and reused when the
    wrapper forwards to the original model.
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
        return tuple(self._tensor_arg_names)

    def forward(self, *tensor_args: torch.Tensor) -> Any:
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
    """
    Singleton TensorBoard logger for training metrics, hyperparameters, and model diagnostics.

    Usage:
        import tb_logger
        tb_logger.set_config(config)
        tb_logger.new_run(save_dir / "tensorboard")

        # During training:
        tb_logger.log_epoch_metrics(epoch, train_loss, train_reg, train_total,
                                 valid_loss, valid_reg, valid_total)
        tb_logger.log_learning_rate(epoch, lr)

        # At the end:
        tb_logger.log_final_metrics({"final_score": score})
        tb_logger.close()
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
        return self._enabled

    def set_config(self, config: Config) -> None:
        self._config = config

    def new_run(self, save_dir: str | Path) -> None:
        """
        Initialize a new TensorBoard run in the given directory.
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
        """
        Flush and close the writer.
        """
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
        self._enabled = False

    def _check_initialized(self) -> None:
        if self._writer is None:
            raise RuntimeError("TensorBoardLogger not initialized. Call new_run() first.")

    @property
    def writer(self) -> SummaryWriter:
        """
        Return the writer, raising if not initialized. Enables type-safe access.
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
        """
        Log all per-epoch training (and optional validation) metrics.
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
        """
        Log the current learning rate.
        """
        if not self._enabled:
            return

        self.writer.add_scalar("other/learning_rate", lr, epoch)

    def log_model_weights(self, epoch: int, model: torch.nn.Module) -> None:
        """
        Log histograms of model parameter values and gradients.
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
        """
        Log the model computation graph (call once at the start of training).
        """
        if not self._enabled:
            return

        try:
            non_tensor_input_names = [
                name for name, value in input_example.items() if not isinstance(value, torch.Tensor)
            ]

            graph_model: torch.nn.Module = model
            graph_inputs = tuple(input_example.values())

            if non_tensor_input_names:
                graph_model = _TensorBoardGraphWrapper(model, input_example)
                graph_inputs = tuple(input_example[name] for name in graph_model.tensor_arg_names)

                if not graph_inputs:
                    logging.warning(
                        "Skipping TensorBoard graph logging because the model has no tensor inputs to trace."
                    )
                    return

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
        """
        Log a single scalar value.
        """
        if not self._enabled:
            return

        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """
        Log a histogram of tensor values.
        """
        if not self._enabled:
            return

        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Log a text string.
        """
        if not self._enabled:
            return

        self.writer.add_text(tag, text, step)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """
        Log a matplotlib figure as an image.
        """
        if not self._enabled:
            return

        self.writer.add_figure(tag, figure, step)

    def log_final_metrics(self, metric_dict: dict[str, float]) -> None:
        """
        Log final run-level metrics alongside hyperparameters using add_hparams.
        Call once at the end of training with the final scores.
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
        """
        Flush pending events to disk.
        """
        if self._writer is not None:
            self._writer.flush()

    @staticmethod
    def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        """
        Flatten a nested dictionary into dot-separated keys.
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
