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

import seaborn as sns
import matplotlib.pyplot as plt
import tqdm as tqdm
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from xanesnet.utils.io import save_xanes, mkdir_output
from xanesnet.utils.mode import Mode
from xanesnet.utils.xanes import XANES


def plot(path: Path, mode: Mode, result, dataset, pred_eval, scheme, metadata):
    if scheme.mh_flag and not pred_eval:
        plot_predict_mh(path, mode, result, dataset, metadata)
    elif scheme.recon_flag:
        plot_predict_recon(path, mode, result, dataset, pred_eval)
    else:
        plot_predict(path, mode, result, dataset, pred_eval)


def _plot_single(save_path, id_, y_pred, y_target=None):
    plt.figure()
    plt.plot(y_pred, label="Prediction")
    if y_target is not None:
        plt.plot(y_target, label="Target")
    plt.legend(loc="upper right")
    plt.savefig(save_path / f"{id_}.pdf")
    plt.close()


def _plot_mean_std(data, label):
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    plt.plot(mean, label=label)
    plt.fill_between(
        np.arange(mean.shape[0]),
        mean + std,
        mean - std,
        alpha=0.4,
        linewidth=0,
    )


def plot_predict(path: Path, mode: Mode, result, dataset, pred_eval):
    save_path = mkdir_output(path, "plot")

    if mode == Mode.XANES_TO_XYZ:
        predict = result.xyz_pred[0]
    elif mode == Mode.XYZ_TO_XANES:
        predict = result.xanes_pred[0]
    else:
        raise ValueError("Unsupported prediction mode.")

    file_names = dataset.file_names
    total_target, total_predict = [], []

    for i, id_ in enumerate(file_names):
        y_pred = predict[i]
        y_target = result.targets[i] if pred_eval else None
        _plot_single(save_path, id_, y_pred, y_target)
        total_predict.append(y_pred)

        if pred_eval:
            total_target.append(y_target)

    total_predict = np.asarray(total_predict)
    total_target = np.asarray(total_target) if pred_eval else None

    # Plot average curves
    sns.set_style("dark")
    plt.figure()
    if pred_eval:
        _plot_mean_std(total_target, "Target")
    _plot_mean_std(total_predict, "Prediction")

    plt.legend(loc="best")
    plt.grid()
    plt.savefig(save_path / "avg_plot.pdf")


def plot_predict_mh(path, mode, result, dataset, metadata):
    """
    Plot prediction results for Multi-Head (MH) schemes.
    Each head is plotted and saved in a separate sub-directory.
    """
    save_path = mkdir_output(path, "plot")

    # Select prediction according to mode
    if mode == Mode.XANES_TO_XYZ:
        predict = result.xyz_pred[0]  # (N, H, M)
    elif mode == Mode.XYZ_TO_XANES:
        predict = result.xanes_pred[0]  # (N, H, M)
    else:
        raise ValueError("Unsupported prediction mode for MH plotting.")

    file_names = dataset.file_names

    nhead = predict.shape[1]
    head_names = metadata["dataset"]["params"]["head_names"]

    for h in range(nhead):
        # ---- per-head directory ----
        head_dir = save_path / head_names[h]
        head_dir.mkdir(parents=True, exist_ok=True)

        total_predict, total_target = [], []

        # ---- per-sample plots ----
        for i, id_ in enumerate(file_names):
            y_pred = predict[i, h]

            _plot_single(
                head_dir,
                id_,
                y_pred,
                None,
            )

            total_predict.append(y_pred)

        total_predict = np.asarray(total_predict)

        # ---- average plot for this head ----
        sns.set_style("dark")
        plt.figure()

        _plot_mean_std(total_predict, label="Prediction")

        plt.legend(loc="best")
        plt.grid()
        plt.title(head_names[h])
        plt.savefig(head_dir / "avg_plot.pdf")
        plt.close()


def plot_predict_recon(path: Path, mode, result, dataset, pred_eval):
    save_path = mkdir_output(path, "plot")

    feat = np.array([data.x.cpu().numpy() for data in dataset])
    target = np.array([data.y.cpu().numpy() for data in dataset]) if pred_eval else None

    file_names = dataset.file_names

    if mode is Mode.XANES_TO_XYZ:
        if not pred_eval:
            for id_, xanes_, xanes_recon_, xyz_pred_ in tqdm.tqdm(
                zip(file_names, feat, result.xanes_recon[0], result.xyz_pred[0])
            ):
                sns.set()
                fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))

                ax1.plot(xanes_recon_, label="Reconstruction")
                ax1.set_title(f"Spectrum Reconstruction")
                ax1.plot(xanes_, label="Target")
                ax1.legend(loc="upper left")

                ax2.plot(xyz_pred_, label="Prediction")
                ax2.set_title(f"Structure Prediction")
                ax2.legend(loc="upper left")

                plt.savefig(save_path / f"{id_}.pdf")
                fig.clf()
                plt.close(fig)

        else:
            for id_, xyz_, xanes_, xanes_recon_, xyz_pred_ in tqdm.tqdm(
                zip(file_names, target, feat, result.xanes_recon[0], result.xyz_pred[0])
            ):
                sns.set()
                fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))

                ax1.plot(xanes_recon_, label="Reconstruction")
                ax1.set_title(f"Spectrum Reconstruction")
                ax1.plot(xanes_, label="target")
                ax1.legend(loc="upper left")

                ax2.plot(xyz_pred_, label="Prediction")
                ax2.set_title(f"Structure Prediction")
                ax2.plot(xyz_, label="target")
                ax2.legend(loc="upper left")

                plt.savefig(save_path / f"{id_}.pdf")
                fig.clf()
                plt.close(fig)

    if mode is Mode.XYZ_TO_XANES:
        if not pred_eval:
            for id_, xyz_, xyz_recon_, xanes_pred_ in tqdm.tqdm(
                zip(file_names, feat, result.xyz_recon[0], result.xanes_pred[0])
            ):
                sns.set()
                fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))

                ax1.plot(xyz_recon_, label="Reconstruction")
                ax1.set_title(f"Structure Reconstruction")
                ax1.plot(xyz_, label="target")
                ax1.legend(loc="upper left")

                ax2.plot(xanes_pred_, label="Prediction")
                ax2.set_title(f"Spectrum Prediction")
                ax2.legend(loc="upper left")

                plt.savefig(save_path / f"{id_}.pdf")
                fig.clf()
                plt.close(fig)
        else:
            for id_, xyz_, xanes_, xyz_recon_, xanes_pred_ in tqdm.tqdm(
                zip(file_names, feat, target, result.xyz_recon[0], result.xanes_pred[0])
            ):
                sns.set()
                fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))
                ax1.plot(xyz_recon_, label="Reconstruction")
                ax1.set_title(f"Structure Reconstruction")
                ax1.plot(xyz_, label="Target")
                ax1.legend(loc="upper left")

                ax2.plot(xanes_pred_, label="Prediction")
                ax2.set_title(f"Spectrum Prediction")
                ax2.plot(xanes_, label="Target")
                ax2.legend(loc="upper left")

                plt.savefig(save_path / f"{id_}.pdf")
                fig.clf()
                plt.close(fig)

    elif mode is Mode.BIDIRECTIONAL:
        for (
            id_,
            xyz_,
            xanes_,
            xyz_recon_,
            xanes_recon_,
            xyz_pred_,
            xanes_pred_,
        ) in tqdm.tqdm(
            zip(
                file_names,
                feat,
                target,
                result.xyz_recon[0],
                result.xanes_recon[0],
                result.xyz_pred[0],
                result.xanes_pred[0],
            )
        ):
            sns.set()
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(20, 20))

            ax1.plot(xyz_recon_, label="Reconstruction")
            ax1.set_title(f"Structure Reconstruction")
            ax1.plot(xyz_, label="target")
            ax1.legend(loc="upper left")

            ax2.plot(xanes_recon_, label="Reconstruction")
            ax2.set_title(f"Spectrum Reconstruction")
            ax2.plot(xanes_, label="target")
            ax2.legend(loc="upper left")

            ax3.plot(xanes_pred_, label="Prediction")
            ax3.set_title(f"Spectrum Prediction")
            ax3.plot(xanes_, label="target")
            ax3.legend(loc="upper left")

            ax4.plot(xyz_pred_, label="Prediction")
            ax4.set_title(f"Structure Prediction")
            ax4.plot(xyz_, label="target")
            ax4.legend(loc="upper left")

            plt.savefig(save_path / f"{id_}.pdf")
            fig.clf()
            plt.close(fig)
