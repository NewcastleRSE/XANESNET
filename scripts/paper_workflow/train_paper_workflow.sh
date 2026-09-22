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

# With no arguments, run every training. Otherwise, arguments select the numbered trainings to run.
runs=("$@")
if [[ ${#runs[@]} -eq 0 ]]; then
    runs=(1 2)
fi

for run in "${runs[@]}"; do
    case "$run" in
        1 | 2)
            ;;
        -h | --help)
            echo "Usage: $0 [1] [2]"
            echo "Run both training jobs when no run numbers are provided."
            exit 0
            ;;
        *)
            echo "ERROR: unknown training number '$run'. Choose 1 or 2." >&2
            exit 2
            ;;
    esac
done

run_training() {
    local run="$1"

    case "$run" in
        1)
            echo "[1/2] train: configs/paper_workflow/mlp_train.yaml"
            python -m xanesnet.cli train \
                -i ./configs/paper_workflow/mlp_train.yaml \
                -o "$OUT_DIR" \
                -n "$MLP_RUN_NAME" \
                --yes
            ;;
        2)
            echo "[2/2] train: configs/paper_workflow/schnet_train.yaml"
            python -m xanesnet.cli train \
                -i ./configs/paper_workflow/schnet_train.yaml \
                -o "$OUT_DIR" \
                -n "$SCHNET_RUN_NAME" \
                --yes
            ;;
    esac
}

for run in "${runs[@]}"; do
    run_training "$run"
done

echo "Training finished. Results are under: ${OUT_DIR}"
