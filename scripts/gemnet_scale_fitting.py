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

"""Fit GemNet scale factors from forward-pass variance statistics."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

import torch

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.datasets import DatasetRegistry
from xanesnet.datasources import DataSourceRegistry
from xanesnet.models import ModelRegistry
from xanesnet.models.gemnet.layers.scaling import ScaleFactor as GemNetScaleFactor
from xanesnet.models.gemnet_oc.layers.scaling import ScaleFactor as GemNetOcScaleFactor
from xanesnet.serialization.config import Config, load_raw_config, validate_config_train
from xanesnet.serialization.splits import save_split_indices
from xanesnet.utils.logger import setup_logging
from xanesnet.utils.random import set_global_seed

HookFunction: TypeAlias = Callable[[torch.nn.Module, tuple[Any, ...], dict[str, Any], Any], None]


class _BatchProtocol(Protocol):
    """Minimal batch interface used by the scale-fitting forward pass."""

    def to(self, device: torch.device) -> Any:
        """Move contained tensors to ``device``."""
        ...


class _BatchProcessorProtocol(Protocol):
    """Minimal batchprocessor interface used to prepare model inputs."""

    def input_preparation(self, batch: _BatchProtocol) -> dict[str, Any]:
        """Return keyword arguments consumed by the model forward call."""
        ...


class _DatasetProtocol(Protocol):
    """Minimal dataset interface required for scale fitting."""

    dataset_type: str
    train_subset: Any
    collate_fn: Callable[..., Any]

    def prepare(self) -> None:
        """Prepare dataset samples and cached graph inputs."""
        ...

    def setup_splits(self) -> None:
        """Populate train, validation, and test subsets."""
        ...

    def check_preload(self) -> None:
        """Validate or build preloaded sample state."""
        ...

    def get_dataloader(self) -> type:
        """Return the dataset-specific dataloader class."""
        ...

    def get_all_subset_indices(self) -> list[list[int]]:
        """Return ordered split-index lists."""
        ...


class _DataLoaderProtocol(Protocol):
    """Iterable dataloader interface used by the fitter."""

    def __iter__(self) -> Iterator[_BatchProtocol]:
        """Iterate batches."""
        ...


class _ScaleFactorProtocol(Protocol):
    """Scale-factor module surface used by this script."""

    scale_factor: torch.Tensor

    def register_forward_hook(
        self,
        hook: HookFunction,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a PyTorch forward hook and return its removable handle."""
        ...


_SCALE_FACTOR_TYPES: tuple[type[torch.nn.Module], ...] = (GemNetScaleFactor, GemNetOcScaleFactor)


def _collect_scale_factors(model: torch.nn.Module) -> dict[str, _ScaleFactorProtocol]:
    """Collect GemNet scale-factor submodules by module path.

    Args:
        model: GemNet or GemNet-OC model to inspect.

    Returns:
        Mapping from module path to scale-factor module.
    """

    return {
        name: cast(_ScaleFactorProtocol, m) for name, m in model.named_modules() if isinstance(m, _SCALE_FACTOR_TYPES)
    }


def _build_dataset(config: Config) -> _DatasetProtocol:
    """Instantiate and prepare the configured dataset.

    Args:
        config: Validated training configuration.

    Returns:
        Prepared dataset object with train/test split metadata.
    """

    ds_cfg = config.section("datasource")
    datasource = DataSourceRegistry.get(ds_cfg.get_str("datasource_type"))(**ds_cfg.as_kwargs())

    dset_cfg = config.section("dataset")
    dataset = cast(
        _DatasetProtocol,
        DatasetRegistry.get(dset_cfg.get_str("dataset_type"))(**dset_cfg.as_kwargs(), datasource=datasource),
    )
    dataset.prepare()
    dataset.setup_splits()
    dataset.check_preload()
    return dataset


def _build_model(config: Config) -> torch.nn.Module:
    """Instantiate a GemNet model with scale-file loading disabled.

    Args:
        config: Validated training configuration.

    Returns:
        Configured GemNet or GemNet-OC model.

    Raises:
        ValueError: If the configured model type is not supported by the fitter.
    """

    model_cfg = config.section("model")
    model_type = model_cfg.get_str("model_type")
    if model_type not in ("gemnet", "gemnet_oc"):
        raise ValueError(f"Scale fitting is only meaningful for gemnet / gemnet_oc; got {model_type!r}")
    kwargs: dict[str, Any] = model_cfg.as_kwargs()
    kwargs["scale_file"] = None
    model = cast(torch.nn.Module, ModelRegistry.get(model_type)(**kwargs))
    return model


def _build_dataloader(config: Config, dataset: _DatasetProtocol) -> _DataLoaderProtocol:
    """Build the training dataloader used for variance observations.

    Args:
        config: Validated training configuration.
        dataset: Prepared dataset with a non-empty training subset.

    Returns:
        Iterable dataloader over training batches.

    Raises:
        ValueError: If the dataset has no training subset.
    """

    trainer_cfg = config.section("trainer")
    batch_size = trainer_cfg.get_int("batch_size")
    num_workers = trainer_cfg.get_int("num_workers")
    shuffle = trainer_cfg.get_bool("shuffle")
    drop_last = trainer_cfg.get_bool("drop_last")

    if dataset.train_subset is None:
        raise ValueError("Training subset is required for scale fitting.")

    dataloader_cls = dataset.get_dataloader()
    return cast(
        _DataLoaderProtocol,
        dataloader_cls(
            dataset.train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False if num_workers == 0 else True,
            prefetch_factor=None if num_workers == 0 else 2,
        ),
    )


def _default_split_indexfile_path(scales_output_path: Path) -> Path:
    """Return the default split-index path next to the scale JSON.

    Args:
        scales_output_path: Path where fitted scale factors will be written.

    Returns:
        Sibling ``split_indices.json`` path.
    """

    return scales_output_path.with_name("split_indices.json")


class _Observer:
    """Accumulate variance observations for one scale-factor module.

    Attributes:
        var_in_sum: Sum of reference variances over hook calls.
        var_out_sum: Sum of module-input variances over hook calls.
        n: Number of hook calls observed.
        first_call_index: Reserved diagnostic field for call-order metadata.
    """

    def __init__(self) -> None:
        """Initialize empty variance accumulators."""

        self.var_in_sum = 0.0
        self.var_out_sum = 0.0
        self.n = 0
        self.first_call_index: int | None = None

    def __call__(self, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        """Observe one ``ScaleFactor.forward(x, ref=None)`` call.

        Args:
            module: Scale-factor module that emitted the hook.
            args: Positional forward arguments.
            kwargs: Keyword forward arguments.
            output: Forward output, unused by this observer.
        """

        x = args[0] if len(args) >= 1 else kwargs["x"]
        if len(args) >= 2:
            ref = args[1]
        else:
            ref = kwargs.get("ref")

        var_out = float(x.detach().float().var(unbiased=False).item())
        if ref is not None:
            var_in = float(ref.detach().float().var(unbiased=False).item())
        else:
            var_in = var_out

        self.var_in_sum += var_in
        self.var_out_sum += var_out
        self.n += 1


class _OrderRecorder:
    """Record the first global hook-call index for one scale-factor module.

    Attributes:
        idx: First invocation index, or ``None`` when the module was not reached.
    """

    def __init__(self) -> None:
        """Initialize the recorder with no observed call index."""

        self.idx: int | None = None

    def make(self, counter: list[int]) -> HookFunction:
        """Create a forward hook bound to a shared call-order counter.

        Args:
            counter: Single-item mutable counter incremented on first observation.

        Returns:
            PyTorch forward hook function.
        """

        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            """Store the first global invocation index for this recorder."""

            if self.idx is None:
                self.idx = counter[0]
                counter[0] += 1

        return hook


def _run_forward(
    model: torch.nn.Module,
    dataloader_iter_box: list[Iterator[_BatchProtocol]],
    dataloader: Iterable[_BatchProtocol],
    batchprocessor: _BatchProcessorProtocol,
    device: torch.device,
) -> None:
    """Run one no-gradient model forward on the next training batch.

    Args:
        model: Model to execute.
        dataloader_iter_box: Mutable single-item holder for the active iterator.
        dataloader: Iterable used to refresh the iterator after exhaustion.
        batchprocessor: Component that converts batches to model keyword inputs.
        device: Target device for the batch.
    """

    try:
        batch = next(dataloader_iter_box[0])
    except StopIteration:
        dataloader_iter_box[0] = iter(dataloader)
        batch = next(dataloader_iter_box[0])
    batch.to(device)
    inputs = batchprocessor.input_preparation(batch)
    _ = model(**inputs)


def fit_scales(
    model: torch.nn.Module,
    dataloader: Iterable[_BatchProtocol],
    batchprocessor: _BatchProcessorProtocol,
    device: torch.device,
    num_batches: int,
) -> dict[str, float]:
    """Fit scale factors by matching input and output variances.

    Args:
        model: GemNet or GemNet-OC model containing scale-factor modules.
        dataloader: Iterable over training batches.
        batchprocessor: Component that prepares model inputs from each batch.
        device: Target device for forward passes.
        num_batches: Number of forward passes observed per scale factor.

    Returns:
        Mapping from scale-factor module path to fitted scalar value.

    Raises:
        RuntimeError: If the model contains no scale-factor modules.
    """

    factors = _collect_scale_factors(model)
    if not factors:
        raise RuntimeError("Model contains no ScaleFactor submodules; nothing to fit.")

    with torch.no_grad():
        for sf in factors.values():
            sf.scale_factor.fill_(1.0)

    model.eval().to(device)
    dataloader_iter_box = [iter(dataloader)]

    counter = [0]
    recorders: dict[str, _OrderRecorder] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, sf in factors.items():
        rec = _OrderRecorder()
        recorders[name] = rec
        handles.append(sf.register_forward_hook(rec.make(counter), with_kwargs=True))

    with torch.no_grad():
        _run_forward(model, dataloader_iter_box, dataloader, batchprocessor, device)

    for h in handles:
        h.remove()

    unreached = [n for n, r in recorders.items() if r.idx is None]
    if unreached:
        logging.warning(
            "%d ScaleFactors not reached on dry-run forward (will be left at 1.0 -> identity): %s",
            len(unreached),
            unreached,
        )

    sorted_names = sorted(
        (n for n, r in recorders.items() if r.idx is not None),
        key=lambda n: recorders[n].idx,  # type: ignore[arg-type, return-value]
    )

    logging.info("Found %d ScaleFactors; fitting %d in forward order.", len(factors), len(sorted_names))

    fitted: dict[str, float] = {}
    for i, name in enumerate(sorted_names):
        sf = factors[name]
        observer = _Observer()
        handle = sf.register_forward_hook(observer, with_kwargs=True)

        with torch.no_grad():
            for _ in range(num_batches):
                _run_forward(model, dataloader_iter_box, dataloader, batchprocessor, device)

        handle.remove()

        if observer.n == 0 or observer.var_out_sum <= 0.0:
            logging.warning(
                "[%d/%d] %s: no valid samples (n=%d, var_out=%.3e); leaving unfitted.",
                i + 1,
                len(sorted_names),
                name,
                observer.n,
                observer.var_out_sum,
            )
            continue

        var_in_mean = observer.var_in_sum / observer.n
        var_out_mean = observer.var_out_sum / observer.n
        if var_out_mean <= 0.0 or not math.isfinite(var_out_mean):
            logging.warning(
                "[%d/%d] %s: degenerate var_out=%.3e; leaving unfitted.", i + 1, len(sorted_names), name, var_out_mean
            )
            continue

        scale = math.sqrt(max(var_in_mean, 1e-12) / max(var_out_mean, 1e-12))
        with torch.no_grad():
            sf.scale_factor.fill_(scale)
        fitted[name] = scale
        logging.info(
            "[%d/%d] %-60s var_in=%.3e var_out=%.3e -> scale=%.4f",
            i + 1,
            len(sorted_names),
            name,
            var_in_mean,
            var_out_mean,
            scale,
        )

    return fitted


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments without the executable name.

    Returns:
        Parsed namespace for scale fitting.
    """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--config", type=str, required=True, help="Path to a training YAML config.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to write the fitted scale-factor JSON.")
    parser.add_argument(
        "--split-indexfile-output",
        type=str,
        default=None,
        help="Path to write the split-index JSON (default: next to --output as split_indices.json).",
    )
    parser.add_argument("--num-batches", type=int, default=16, help="Forward passes per ScaleFactor (default: 16).")
    parser.add_argument("--device", type=str, default=None, help="Override device (default: from config).")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed (default: from config).")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    """Run the GemNet scale-fitting command-line interface.

    Args:
        argv: Command-line arguments without the executable name.
    """

    setup_logging(logging.INFO)
    args = parse_args(argv)

    logging.info("Loading config: %s", args.config)
    config_raw = load_raw_config(args.config)
    config = validate_config_train(config_raw)

    seed = args.seed if args.seed is not None else config.get_optional_int("seed")
    seed = set_global_seed(seed)
    logging.info("Global seed: %d", seed)

    device_str = args.device if args.device is not None else config.get_str("device")
    device = torch.device(device_str)
    logging.info("Using device: %s", device)

    dataset = _build_dataset(config)
    model = _build_model(config)
    dataloader = _build_dataloader(config, dataset)
    batchprocessor = cast(_BatchProcessorProtocol, BatchProcessorRegistry.get(dataset.dataset_type, model.model_type)())

    logging.info("Fitting scale factors over %d batches each.", args.num_batches)
    fitted = fit_scales(model, dataloader, batchprocessor, device, args.num_batches)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fitted, f, indent=2, sort_keys=True)
    logging.info("Wrote %d fitted scale factors to %s", len(fitted), out_path)

    split_indexfile_path = (
        Path(args.split_indexfile_output)
        if args.split_indexfile_output is not None
        else _default_split_indexfile_path(out_path)
    )
    save_split_indices(split_indexfile_path, dataset.get_all_subset_indices())
    logging.info("Wrote split indices for reuse during training to %s", split_indexfile_path)


if __name__ == "__main__":
    main(sys.argv[1:])
