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

import argparse
import json
import logging
import math
import sys
from pathlib import Path

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

# The two GemNet variants ship independent (but equivalent) ScaleFactor
# implementations — the fitter handles either via a tuple isinstance check.
_SCALE_FACTOR_TYPES: tuple[type[torch.nn.Module], ...] = (GemNetScaleFactor, GemNetOcScaleFactor)


def _collect_scale_factors(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return {name: m for name, m in model.named_modules() if isinstance(m, _SCALE_FACTOR_TYPES)}


###############################################################################
############################### small helpers #################################
###############################################################################


def _build_dataset(config: Config):
    ds_cfg = config.section("datasource")
    datasource = DataSourceRegistry.get(ds_cfg.get_str("datasource_type"))(**ds_cfg.as_kwargs())

    dset_cfg = config.section("dataset")
    dataset = DatasetRegistry.get(dset_cfg.get_str("dataset_type"))(**dset_cfg.as_kwargs(), datasource=datasource)
    dataset.prepare()
    dataset.setup_splits()
    dataset.check_preload()
    return dataset


def _build_model(config: Config) -> torch.nn.Module:
    model_cfg = config.section("model")
    model_type = model_cfg.get_str("model_type")
    if model_type not in ("gemnet", "gemnet_oc"):
        raise ValueError(f"Scale fitting is only meaningful for gemnet / gemnet_oc; got {model_type!r}")
    # Force scale_file=None so the model starts with un-fitted (zero) factors.
    kwargs = model_cfg.as_kwargs()
    kwargs["scale_file"] = None
    model = ModelRegistry.get(model_type)(**kwargs)
    return model


def _build_dataloader(config: Config, dataset):
    trainer_cfg = config.section("trainer")
    batch_size = trainer_cfg.get_int("batch_size")
    num_workers = trainer_cfg.get_int("num_workers")
    shuffle = trainer_cfg.get_bool("shuffle")
    drop_last = trainer_cfg.get_bool("drop_last")

    if dataset.train_subset is None:
        raise ValueError("Training subset is required for scale fitting.")

    dataloader_cls = dataset.get_dataloader()
    return dataloader_cls(
        dataset.train_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False if num_workers == 0 else True,
        prefetch_factor=None if num_workers == 0 else 2,
    )


def _default_split_indexfile_path(scales_output_path: Path) -> Path:
    return scales_output_path.with_name("split_indices.json")


###############################################################################
############################### FORWARD HOOK  #################################
###############################################################################


class _Observer:
    """
    Forward hook that accumulates ``var_in`` and ``var_out`` for one ScaleFactor.
    """

    def __init__(self) -> None:
        self.var_in_sum = 0.0
        self.var_out_sum = 0.0
        self.n = 0
        self.first_call_index: int | None = None

    def __call__(self, module: torch.nn.Module, args, kwargs, output) -> None:
        # Forward signature: forward(x, ref=None)
        x = args[0] if len(args) >= 1 else kwargs["x"]
        if len(args) >= 2:
            ref = args[1]
        else:
            ref = kwargs.get("ref")

        var_out = float(x.detach().float().var(unbiased=False).item())
        if ref is not None:
            var_in = float(ref.detach().float().var(unbiased=False).item())
        else:
            # No reference: degenerate observation; var_in == var_out gives scale 1.
            var_in = var_out

        self.var_in_sum += var_in
        self.var_out_sum += var_out
        self.n += 1


class _OrderRecorder:
    """
    Tiny hook that records the global call index of the first invocation.
    """

    def __init__(self) -> None:
        self.idx: int | None = None

    def make(self, counter: list[int]):
        def hook(module, args, kwargs, output) -> None:
            if self.idx is None:
                self.idx = counter[0]
                counter[0] += 1

        return hook


def _run_forward(model, dataloader_iter_box, dataloader, batchprocessor, device) -> None:
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
    dataloader,
    batchprocessor,
    device: torch.device,
    num_batches: int,
) -> dict[str, float]:
    factors = _collect_scale_factors(model)
    if not factors:
        raise RuntimeError("Model contains no ScaleFactor submodules; nothing to fit.")

    # Reset all factors to 1.0 (identity) so the dry-run / observation
    # passes see the un-scaled forward variances. Anything not visited
    # downstream of the dry run will stay at 1.0 in the final model.
    with torch.no_grad():
        for sf in factors.values():
            sf.scale_factor.fill_(1.0)

    model.eval().to(device)
    dataloader_iter_box = [iter(dataloader)]

    # Phase 1: discover the order in which ScaleFactors are first hit
    counter = [0]
    recorders: dict[str, _OrderRecorder] = {}
    handles = []
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

    # Phase 2: fit one ScaleFactor at a time
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

    # Anything still unreachable stays at 0.0 (pass-through), and we just
    # don't include it in the output JSON.
    return fitted


###############################################################################
#################################### CLI ######################################
###############################################################################


def parse_args(argv: list[str]) -> argparse.Namespace:
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
    batchprocessor = BatchProcessorRegistry.get(dataset.dataset_type, model.model_type)()

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
