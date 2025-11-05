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

from xanesnet.creator import create_predict_scheme, create_dataset
from xanesnet.utils.mode import get_mode, Mode
from xanesnet.utils.plot import plot
from xanesnet.utils.io import (
    save_predict_result,
    load_descriptors_from_local,
    load_models_from_local,
    load_model_from_local,
)

# from xanesnet.post_shap import shap_analysis, shap_analysis_gnn


def predict(config, args, metadata):
    """
    Prediction pipeline for non-GNN models.
    """
    logging.info(f">> Prediction mode: {args.mode}")
    mode = get_mode(args.mode)

    model_path = Path(args.in_model)
    root_path = config["dataset"].get("root_path")
    xyz_path = config["dataset"].get("xyz_path", None)
    xanes_path = config["dataset"].get("xanes_path", None)

    _verify_mode(xyz_path, xanes_path, metadata, mode)

    # Load descriptor list
    descriptor_list = load_descriptors_from_local(model_path)

    # Enable model evaluation if test data is present
    pred_eval = xyz_path is not None and xanes_path is not None

    # Load, encode, and preprocess data
    dataset = _setup_datasets(
        root_path, xyz_path, xanes_path, metadata, mode, descriptor_list
    )

    # Setup prediction scheme
    scheme = _setup_scheme(dataset, mode, metadata, pred_eval)

    # Predict with loaded models and scheme
    saved_train_scheme = metadata["scheme"]
    result = _run_prediction(scheme, model_path, saved_train_scheme)

    # Set output path
    path = Path("outputs") / args.in_model
    path = Path(str(path).replace("models/", ""))

    # Save raw prediction result
    if config["result_save"]:
        save_predict_result(path, mode, result, dataset, scheme.recon_flag)

    # Plot prediction result
    if config["plot_save"]:
        plot(path, mode, result, dataset, pred_eval, scheme.recon_flag)

    logging.info("\nPrediction results saved to disk: %s", path.resolve().as_uri())

    # SHAP analysis
    # if config["shap"] and predict_scheme == "std":
    #     xyz_data = scheme.xyz_data
    #     nsamples = config["shap_params"]["nsamples"]
    #     shap_analysis_gnn(path, mode, model, index, xyz_data, xanes_data, nsamples)


def _setup_datasets(root_path, xyz_path, xanes_path, metadata, mode, descriptor_list):
    dataset_type = metadata["dataset"]["type"]
    logging.info(">> Initialising prediction datasets...")

    # Pack kwargs
    kwargs = {
        "root": root_path,
        "xyz_path": xyz_path,
        "xanes_path": xanes_path,
        "mode": mode,
        "descriptors": descriptor_list,
        **metadata["dataset"]["params"],
    }

    dataset = create_dataset(dataset_type, **kwargs)

    logging.info(
        f">> Dataset Summary: # of samples = {len(dataset)}, feature size = {dataset.x_size}, label(y) size = {dataset.y_size}"
    )

    return dataset


def _setup_scheme(dataset, mode, metadata, pred_eval):
    model_type = metadata["model"]["type"]

    kwargs = {
        "pred_mode": mode,
        "pred_eval": pred_eval,
        "scaler": metadata["standardscaler"],
        **metadata["dataset"]["params"],
    }

    scheme = create_predict_scheme(model_type, dataset, mode, **kwargs)
    return scheme


def _run_prediction(scheme, model_dir: Path, predict_scheme: str):
    """
    Loads models and runs the prediction based on the specified scheme.
    """
    if predict_scheme == "bootstrap":
        if "bootstrap" not in str(model_dir):
            raise ValueError("Invalid bootstrap directory")

        model_list = load_models_from_local(model_dir)
        result = scheme.predict_bootstrap(model_list)

    elif predict_scheme == "ensemble":
        if "ensemble" not in str(model_dir):
            raise ValueError("Invalid ensemble directory")
        model_list = load_models_from_local(model_dir)
        result = scheme.predict_ensemble(model_list)

    elif predict_scheme == "std" or predict_scheme == "kfold":
        model = load_model_from_local(model_dir)
        result = scheme.predict_std(model)

    else:
        raise ValueError("Unsupported prediction scheme.")

    return result


def _verify_mode(xyz_path, xanes_path, metadata, mode):
    """
    Checks for consistency between training mode and prediction mode/data.
    """
    train_mode = get_mode(metadata["mode"])

    # inconsistent if
    if train_mode is not mode and mode in {Mode.XYZ_TO_XANES, Mode.XANES_TO_XYZ}:
        raise ValueError(
            f"Inconsistent prediction mode in training ({train_mode}) and prediction ({mode})"
        )

    if (mode is Mode.XYZ_TO_XANES and xyz_path is None) or (
        mode is Mode.XANES_TO_XYZ and xanes_path is None
    ):
        raise ValueError(f"Cannot find prediction dataset.")
