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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# Output directory for dry-run results.
OUT_DIR="./runs/dry_runs"

# Set to "true" to stop on first failure.
STOP_ON_FAILURE=false

TRAIN_CONFIGS=(
    "mlp.yaml"
    "mh_mlp.yaml"
    "mh_mlp_inverse.yaml"
    "mh_cnn.yaml"
    "mh_cnn_inverse.yaml"
    "schnet.yaml"
    "dimenet.yaml"
    "dimenet_pp.yaml"
    "e3ee.yaml"
    "e3ee_full.yaml"
    "envembed.yaml"
    "gemnet.yaml"
    "gemnet_oc.yaml"
    "mlp_deep_ensemble.yaml"
    "mlp_inverse.yaml"
)

mkdir -p "$OUT_DIR"

PASSED=()
FAILED=()

for config in "${TRAIN_CONFIGS[@]}"; do
    config_path="./configs/${config}"
    config_name="${config%.yaml}"
    run_name="dry_${config_name}"

    # extract the processed-data directory from the config
    data_dir=$(grep -E '^\s*root:' "$config_path" 2>/dev/null \
        | head -1 \
        | sed 's/.*root:\s*//' \
        | sed 's/\s*#.*//' \
        | xargs || true)

    echo "============================================================"
    echo "  DRY-RUN: ${config}"
    echo "  Run:     ${run_name}"
    echo "============================================================"

    args=(
        "train"
        "-i" "$config_path"
        "-n" "$run_name"
        "-o" "$OUT_DIR"
        "--yes"
        "--dry-run"
    )

    if python -m xanesnet.cli "${args[@]}"; then
        echo "  PASSED: ${config}"
        PASSED+=("$config")
    else
        echo "  FAILED: ${config}"
        FAILED+=("$config")
        if [[ "$STOP_ON_FAILURE" == "true" ]]; then
            echo "STOP_ON_FAILURE is set; aborting."
            break
        fi
    fi

    # clean up processed data
    if [[ -n "$data_dir" && "$data_dir" == ./data/processed/* ]]; then
        echo "  CLEAN: removing ${data_dir}"
        rm -rf "$data_dir"
    fi

    echo ""
done

echo "============================================================"
echo "  DRY-RUN SUMMARY"
echo "============================================================"
echo "  Passed:  ${#PASSED[@]}"
for c in "${PASSED[@]}"; do
    echo "    OK   $c"
done
echo ""
echo "  Failed:  ${#FAILED[@]}"
for c in "${FAILED[@]}"; do
    echo "    FAIL $c"
done
echo ""
echo "Results saved to: ${OUT_DIR}/"
