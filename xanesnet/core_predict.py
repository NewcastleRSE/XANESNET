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

from pathlib import Path
from typing import Dict

from xanesnet.creator import create_predict_scheme, create_dataset
from xanesnet.utils.mode import get_mode, Mode
from xanesnet.utils.plot import plot
from xanesnet.utils.io import (
    save_predict_result,
    load_descriptors_from_local,
    load_models_from_local,
    load_model_from_local,
)


def predict(config: Dict, args, metadata: Dict):
    """
    Main prediction entry
    """
    logging.info(f">> Prediction mode: {args.mode}")
    mode = get_mode(args.mode)

    model_dir = Path(args.in_model)
    dataset_cfg = config["dataset"]

    root_path = dataset_cfg.get("root_path")
    xyz_path = dataset_cfg.get("xyz_path")
    xanes_path = dataset_cfg.get("xanes_path")

    _verify_mode_consistency(xyz_path, xanes_path, metadata, mode)

    # Load descriptor list
    descriptors = load_descriptors_from_local(model_dir)
    pred_eval = xyz_path is not None and xanes_path is not None

    # Load, encode, and preprocess data
    dataset = setup_datasets(
        root_path, xyz_path, xanes_path, metadata, mode, descriptors
    )

    # Setup prediction scheme
    scheme = setup_scheme(dataset, mode, metadata, pred_eval)

    # Predict with loaded models and scheme
    result = run_prediction(scheme, model_dir, metadata["scheme"])

    # Set output path
    output_path = Path("outputs") / Path(args.in_model).relative_to("models")

    # Save raw prediction result
    if config.get("result_save", True):
        save_predict_result(
            output_path, mode, result, dataset, pred_eval, scheme, metadata
        )

    # Plot prediction result
    if config.get("plot_save", True):
        plot(output_path, mode, result, dataset, pred_eval, scheme, metadata)

    logging.info("Prediction results saved to disk: %s", output_path.resolve().as_uri())


def setup_datasets(root_path, xyz_path, xanes_path, metadata, mode, descriptors):
    """Initialise prediction dataset."""
    dataset_type = metadata["dataset"]["type"]
    logging.info(">> Initialising prediction dataset: %s", dataset_type)

    dataset = create_dataset(
        dataset_type,
        root=root_path,
        xyz_path=xyz_path,
        xanes_path=xanes_path,
        mode=mode,
        descriptors=descriptors,
        **metadata["dataset"]["params"],
    )

    logging.info(
        ">> Dataset summary: samples=%d | X=%s | y=%s",
        len(dataset),
        dataset.x_shape,
        dataset.y_shape,
    )

    return dataset


def setup_scheme(dataset, mode: Mode, metadata: Dict, pred_eval: bool):
    """Initialise prediction scheme."""
    model_type = metadata["model"]["type"]

    return create_predict_scheme(
        model_type,
        dataset,
        pred_mode=mode,
        pred_eval=pred_eval,
        **metadata["dataset"]["params"],
    )


def run_prediction(scheme, model_dir: Path, scheme_type: str):
    """
    Load model(s) and run prediction.
    """
    if scheme_type in {"bootstrap", "ensemble"}:
        _validate_model_dir(model_dir, scheme_type)
        models = load_models_from_local(model_dir)

        if scheme_type == "bootstrap":
            return scheme.predict_bootstrap(models)
        return scheme.predict_ensemble(models)

    if scheme_type in {"std", "kfold"}:
        model = load_model_from_local(model_dir)
        return scheme.predict_std(model)

    raise ValueError(f"Unsupported prediction scheme: {scheme_type}")


def _validate_model_dir(model_dir: Path, scheme_type: str) -> None:
    if scheme_type not in str(model_dir):
        raise ValueError(f"Invalid {scheme_type} model directory: {model_dir}")


def _verify_mode_consistency(xyz_path, xanes_path, metadata, mode):
    """
    Checks for consistency between training mode and prediction mode/data.
    """
    train_mode = get_mode(metadata["mode"])

    if train_mode is not mode and mode in {Mode.XYZ_TO_XANES, Mode.XANES_TO_XYZ}:
        raise ValueError(
            f"Inconsistent prediction mode in training ({train_mode}) and prediction ({mode})"
        )

    if mode is Mode.XYZ_TO_XANES and xyz_path is None:
        raise ValueError(f"Missing XYZ prediction data.")

    if mode is Mode.XANES_TO_XYZ and xanes_path is None:
        raise ValueError(f"Missing XANES prediction data.")
