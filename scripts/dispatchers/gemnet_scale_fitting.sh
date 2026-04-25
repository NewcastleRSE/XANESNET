#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# gemnet_scale_fitting dispatcher
# Edit values below, then run:
#   bash scripts/dispatchers/gemnet_scale_fitting.sh
# -----------------------------------------------------------------------------

# Required
CONFIG="./configs/in_gemnet_oc.yaml"
OUTPUT="./data/scales/scales_gemnet_oc.json"
SPLIT_INDEXFILE_OUTPUT="./data/scales/split_indices_gemnet_oc.json"

# Optional overrides (leave empty to use values from config)
NUM_BATCHES=16
DEVICE="cuda"
SEED=""

args=(
	"scripts/gemnet_scale_fitting.py"
	"--config" "$CONFIG"
	"--output" "$OUTPUT"
	"--split-indexfile-output" "$SPLIT_INDEXFILE_OUTPUT"
	"--num-batches" "$NUM_BATCHES"
)

if [[ -n "$DEVICE" ]]; then
	args+=("--device" "$DEVICE")
fi

if [[ -n "$SEED" ]]; then
	args+=("--seed" "$SEED")
fi

echo "Running: python3 ${args[*]}"
python3 "${args[@]}"
