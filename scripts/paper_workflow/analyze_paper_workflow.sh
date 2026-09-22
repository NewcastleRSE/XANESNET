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

ANALYSIS_COMPARE_NAME="${ANALYSIS_COMPARE_NAME:-paper_workflow_schnet_vs_mlp}"
ANALYSIS_ELEMENTS_NAME="${ANALYSIS_ELEMENTS_NAME:-paper_workflow_schnet_elements}"
ANALYSIS_CLUSTERS_NAME="${ANALYSIS_CLUSTERS_NAME:-paper_workflow_schnet_clusters}"
ANALYSIS_ELEMENT_CLUSTERS_NAME="${ANALYSIS_ELEMENT_CLUSTERS_NAME:-paper_workflow_schnet_element_clusters}"

# With no arguments, run every analysis. Otherwise, arguments select the numbered analyses to run.
runs=("$@")
if [[ ${#runs[@]} -eq 0 ]]; then
    runs=(1 2 3 4)
fi

needs_mlp=false
needs_schnet=false
for run in "${runs[@]}"; do
    case "$run" in
        1)
            needs_mlp=true
            needs_schnet=true
            ;;
        2 | 3 | 4)
            needs_schnet=true
            ;;
        -h | --help)
            echo "Usage: $0 [1] [2] [3] [4]"
            echo "Run all analyses when no run numbers are provided."
            exit 0
            ;;
        *)
            echo "ERROR: unknown analysis number '$run'. Choose from 1, 2, 3, or 4." >&2
            exit 2
            ;;
    esac
done

# Path to the newest inference run for a model run name.
latest_infer_run() {
    local run_name="$1"
    ls -dt "${OUT_DIR}"/infer_"${run_name}"_* 2>/dev/null | head -1 || true
}

if [[ "$needs_mlp" == true ]]; then
    if [[ -z "${MLP_RUN:-}" ]]; then
        MLP_RUN="$(latest_infer_run "$MLP_RUN_NAME")"
    fi
    if [[ ! -d "${MLP_RUN}/predictions" ]]; then
        echo "ERROR: no MLP inference run found. Set MLP_RUN or run infer_paper_workflow.sh first." >&2
        exit 1
    fi
fi

if [[ "$needs_schnet" == true ]]; then
    if [[ -z "${SCHNET_RUN:-}" ]]; then
        SCHNET_RUN="$(latest_infer_run "$SCHNET_RUN_NAME")"
    fi
    if [[ ! -d "${SCHNET_RUN}/predictions" ]]; then
        echo "ERROR: no SchNet inference run found. Set SCHNET_RUN or run infer_paper_workflow.sh first." >&2
        exit 1
    fi
fi

run_analysis() {
    local run="$1"

    case "$run" in
        1)
            echo "[1/4] analyze: configs/paper_workflow/analyze_schnet_vs_mlp.yaml"
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
            echo "[2/4] analyze: configs/paper_workflow/analyze_schnet_elements.yaml"
            python -m xanesnet.cli analyze \
                -i ./configs/paper_workflow/analyze_schnet_elements.yaml \
                -r "$SCHNET_RUN" \
                -d SchNet \
                -o "$OUT_DIR" \
                -n "$ANALYSIS_ELEMENTS_NAME" \
                --yes
            ;;
        3)
            echo "[3/4] analyze: configs/paper_workflow/analyze_schnet_clusters.yaml"
            python -m xanesnet.cli analyze \
                -i ./configs/paper_workflow/analyze_schnet_clusters.yaml \
                -r "$SCHNET_RUN" \
                -d SchNet \
                -o "$OUT_DIR" \
                -n "$ANALYSIS_CLUSTERS_NAME" \
                --yes
            ;;
        4)
            echo "[4/4] analyze: configs/paper_workflow/analyze_schnet_element_clusters.yaml"
            python -m xanesnet.cli analyze \
                -i ./configs/paper_workflow/analyze_schnet_element_clusters.yaml \
                -r "$SCHNET_RUN" \
                -d SchNet \
                -o "$OUT_DIR" \
                -n "$ANALYSIS_ELEMENT_CLUSTERS_NAME" \
                --yes
            ;;
    esac
}

for run in "${runs[@]}"; do
    run_analysis "$run"
done

echo "Analysis finished. Results are under: ${OUT_DIR}"
