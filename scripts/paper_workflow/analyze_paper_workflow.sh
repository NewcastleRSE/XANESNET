#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-./runs/paper_workflow}"
MLP_RUN_NAME="${MLP_RUN_NAME:-paper_workflow_mlp}"
SCHNET_RUN_NAME="${SCHNET_RUN_NAME:-paper_workflow_schnet}"

ANALYSIS_COMPARE_NAME="${ANALYSIS_COMPARE_NAME:-paper_workflow_schnet_vs_mlp}"
ANALYSIS_ELEMENTS_NAME="${ANALYSIS_ELEMENTS_NAME:-paper_workflow_schnet_elements}"
ANALYSIS_CLUSTERS_NAME="${ANALYSIS_CLUSTERS_NAME:-paper_workflow_schnet_clusters}"
ANALYSIS_ELEMENT_CLUSTERS_NAME="${ANALYSIS_ELEMENT_CLUSTERS_NAME:-paper_workflow_schnet_element_clusters}"
ANALYSIS_TENSORBOARD_NAME="${ANALYSIS_TENSORBOARD_NAME:-paper_workflow_tensorboard}"

TEMP_ANALYSIS_CONFIG=""
cleanup_analysis_config() {
    if [[ -n "$TEMP_ANALYSIS_CONFIG" ]]; then
        rm -f -- "$TEMP_ANALYSIS_CONFIG"
    fi
}
trap cleanup_analysis_config EXIT

# With no arguments, run every analysis. Otherwise, arguments select the numbered analyses to run.
runs=("$@")
if [[ ${#runs[@]} -eq 0 ]]; then
    runs=(1 2 3 4 5)
fi

needs_mlp=false
needs_schnet=false
needs_tensorboard=false
for run in "${runs[@]}"; do
    case "$run" in
        1)
            needs_mlp=true
            needs_schnet=true
            ;;
        2 | 3 | 4)
            needs_schnet=true
            ;;
        5)
            needs_mlp=true
            needs_schnet=true
            needs_tensorboard=true
            ;;
        -h | --help)
            echo "Usage: $0 [1] [2] [3] [4] [5]"
            echo "Run all analyses when no run numbers are provided."
            exit 0
            ;;
        *)
            echo "ERROR: unknown analysis number '$run'. Choose from 1, 2, 3, 4, or 5." >&2
            exit 2
            ;;
    esac
done

# Path to the newest inference run for a model run name.
latest_infer_run() {
    local run_name="$1"
    ls -dt "${OUT_DIR}"/infer_"${run_name}"_* 2>/dev/null | head -1 || true
}

latest_train_run() {
    local run_name="$1"
    ls -dt "${OUT_DIR}"/train_"${run_name}"_* 2>/dev/null | head -1 || true
}

create_tensorboard_analysis_config() {
    local mlp_tensorboard_dir="$1"
    local schnet_tensorboard_dir="$2"
    local config_path="$3"

    python - "$mlp_tensorboard_dir" "$schnet_tensorboard_dir" "$config_path" <<'PY'
import sys
from pathlib import Path

import yaml

template_path = Path("configs/paper_workflow/analyze_tensorboard.yaml")
config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
config["custom_modules"][0]["tensorboard_dirs"] = sys.argv[1:3]
Path(sys.argv[3]).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

if [[ "$needs_mlp" == true ]]; then
    if [[ -z "${MLP_RUN:-}" ]]; then
        MLP_RUN="$(latest_infer_run "$MLP_RUN_NAME")"
    fi
    if [[ -z "${MLP_RUN:-}" || ! -d "${MLP_RUN}/predictions" ]]; then
        echo "ERROR: no MLP inference run found. Set MLP_RUN or run infer_paper_workflow.sh first." >&2
        echo "       Expected '${OUT_DIR}/infer_${MLP_RUN_NAME}_<suffix>/' containing predictions/." >&2
        echo "       OUT_DIR defaults to './runs/paper_workflow'; set OUT_DIR or MLP_RUN if it is elsewhere." >&2
        exit 1
    fi
fi

if [[ "$needs_tensorboard" == true ]]; then
    if [[ -z "${MLP_TRAIN_RUN:-}" ]]; then
        MLP_TRAIN_RUN="$(latest_train_run "$MLP_RUN_NAME")"
    fi
    if [[ -z "${MLP_TRAIN_RUN:-}" || ! -d "${MLP_TRAIN_RUN}/tensorboard" ]]; then
        echo "ERROR: no MLP TensorBoard logs found. Set MLP_TRAIN_RUN or run train_paper_workflow.sh 1." >&2
        echo "       Expected '${OUT_DIR}/train_${MLP_RUN_NAME}_<suffix>/tensorboard/'." >&2
        exit 1
    fi
    if [[ -z "${SCHNET_TRAIN_RUN:-}" ]]; then
        SCHNET_TRAIN_RUN="$(latest_train_run "$SCHNET_RUN_NAME")"
    fi
    if [[ -z "${SCHNET_TRAIN_RUN:-}" || ! -d "${SCHNET_TRAIN_RUN}/tensorboard" ]]; then
        echo "ERROR: no SchNet TensorBoard logs found. Set SCHNET_TRAIN_RUN or run train_paper_workflow.sh 2." >&2
        echo "       Expected '${OUT_DIR}/train_${SCHNET_RUN_NAME}_<suffix>/tensorboard/'." >&2
        exit 1
    fi
fi

if [[ "$needs_schnet" == true ]]; then
    if [[ -z "${SCHNET_RUN:-}" ]]; then
        SCHNET_RUN="$(latest_infer_run "$SCHNET_RUN_NAME")"
    fi
    if [[ -z "${SCHNET_RUN:-}" || ! -d "${SCHNET_RUN}/predictions" ]]; then
        echo "ERROR: no SchNet inference run found. Set SCHNET_RUN or run infer_paper_workflow.sh first." >&2
        echo "       Expected '${OUT_DIR}/infer_${SCHNET_RUN_NAME}_<suffix>/' containing predictions/." >&2
        echo "       OUT_DIR defaults to './runs/paper_workflow'; set OUT_DIR or SCHNET_RUN if it is elsewhere." >&2
        exit 1
    fi
fi

run_analysis() {
    local run="$1"

    case "$run" in
        1)
            echo "[1/5] analyze: configs/paper_workflow/analyze_schnet_vs_mlp.yaml"
            python -m xanesnet.cli analyze \
                -i ./configs/paper_workflow/analyze_schnet_vs_mlp.yaml \
                -r "$MLP_RUN" \
                -r "$SCHNET_RUN" \
                -d MLP SchNet \
                -o "$OUT_DIR" \
                -n "$ANALYSIS_COMPARE_NAME" \
                --yes
            ;;
        2)
            echo "[2/5] analyze: configs/paper_workflow/analyze_schnet_elements.yaml"
            python -m xanesnet.cli analyze \
                -i ./configs/paper_workflow/analyze_schnet_elements.yaml \
                -r "$SCHNET_RUN" \
                -d SchNet \
                -o "$OUT_DIR" \
                -n "$ANALYSIS_ELEMENTS_NAME" \
                --yes
            ;;
        3)
            echo "[3/5] analyze: configs/paper_workflow/analyze_schnet_clusters.yaml"
            python -m xanesnet.cli analyze \
                -i ./configs/paper_workflow/analyze_schnet_clusters.yaml \
                -r "$SCHNET_RUN" \
                -d SchNet \
                -o "$OUT_DIR" \
                -n "$ANALYSIS_CLUSTERS_NAME" \
                --yes
            ;;
        4)
            echo "[4/5] analyze: configs/paper_workflow/analyze_schnet_element_clusters.yaml"
            python -m xanesnet.cli analyze \
                -i ./configs/paper_workflow/analyze_schnet_element_clusters.yaml \
                -r "$SCHNET_RUN" \
                -d SchNet \
                -o "$OUT_DIR" \
                -n "$ANALYSIS_ELEMENT_CLUSTERS_NAME" \
                --yes
            ;;
        5)
            TEMP_ANALYSIS_CONFIG="$(mktemp --suffix=.yaml)"
            create_tensorboard_analysis_config \
                "$MLP_TRAIN_RUN/tensorboard" \
                "$SCHNET_TRAIN_RUN/tensorboard" \
                "$TEMP_ANALYSIS_CONFIG"
            echo "[5/5] analyze: configs/paper_workflow/analyze_tensorboard.yaml"
            python -m xanesnet.cli analyze \
                -i "$TEMP_ANALYSIS_CONFIG" \
                -r "$MLP_RUN" \
                -r "$SCHNET_RUN" \
                -d MLP SchNet \
                -o "$OUT_DIR" \
                -n "$ANALYSIS_TENSORBOARD_NAME" \
                --yes
            rm -f -- "$TEMP_ANALYSIS_CONFIG"
            TEMP_ANALYSIS_CONFIG=""
            ;;
    esac
}

for run in "${runs[@]}"; do
    run_analysis "$run"
done

echo "Analysis finished. Results are under: ${OUT_DIR}"
