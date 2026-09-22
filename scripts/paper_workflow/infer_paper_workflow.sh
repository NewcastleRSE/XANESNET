#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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
#   ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-./runs/paper_workflow}"
MLP_RUN_NAME="${MLP_RUN_NAME:-paper_workflow_mlp}"
SCHNET_RUN_NAME="${SCHNET_RUN_NAME:-paper_workflow_schnet}"

# With no arguments, run every inference. Otherwise, arguments select the numbered inferences to run.
runs=("$@")
if [[ ${#runs[@]} -eq 0 ]]; then
    runs=(1 2)
fi

needs_mlp=false
needs_schnet=false
for run in "${runs[@]}"; do
    case "$run" in
        1)
            needs_mlp=true
            ;;
        2)
            needs_schnet=true
            ;;
        -h | --help)
            echo "Usage: $0 [1] [2]"
            echo "Run both inference jobs when no run numbers are provided."
            exit 0
            ;;
        *)
            echo "ERROR: unknown inference number '$run'. Choose 1 or 2." >&2
            exit 2
            ;;
    esac
done

# Path to the newest final checkpoint for a training run name.
latest_checkpoint() {
    local run_name="$1"
    ls -dt "${OUT_DIR}"/train_"${run_name}"_*/models/final.pth 2>/dev/null | head -1 || true
}

if [[ "$needs_mlp" == true ]]; then
    if [[ -z "${MLP_MODEL:-}" ]]; then
        MLP_MODEL="$(latest_checkpoint "$MLP_RUN_NAME")"
    fi
    if [[ ! -f "$MLP_MODEL" ]]; then
        echo "ERROR: no MLP checkpoint found. Set MLP_MODEL or run train_paper_workflow.sh first." >&2
        exit 1
    fi
fi

if [[ "$needs_schnet" == true ]]; then
    if [[ -z "${SCHNET_MODEL:-}" ]]; then
        SCHNET_MODEL="$(latest_checkpoint "$SCHNET_RUN_NAME")"
    fi
    if [[ ! -f "$SCHNET_MODEL" ]]; then
        echo "ERROR: no SchNet checkpoint found. Set SCHNET_MODEL or run train_paper_workflow.sh first." >&2
        exit 1
    fi
fi

run_inference() {
    local run="$1"

    case "$run" in
        1)
            echo "[1/2] infer: configs/paper_workflow/mlp_infer.yaml (model: ${MLP_MODEL})"
            python -m xanesnet.cli infer \
                -i ./configs/paper_workflow/mlp_infer.yaml \
                -m "$MLP_MODEL" \
                -o "$OUT_DIR" \
                -n "$MLP_RUN_NAME" \
                --yes
            ;;
        2)
            echo "[2/2] infer: configs/paper_workflow/schnet_infer.yaml (model: ${SCHNET_MODEL})"
            python -m xanesnet.cli infer \
                -i ./configs/paper_workflow/schnet_infer.yaml \
                -m "$SCHNET_MODEL" \
                -o "$OUT_DIR" \
                -n "$SCHNET_RUN_NAME" \
                --yes
            ;;
    esac
}

for run in "${runs[@]}"; do
    run_inference "$run"
done

echo "Inference finished. Results are under: ${OUT_DIR}"
