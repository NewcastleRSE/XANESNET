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
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Required
CONFIG="./configs/gemnet_oc.yaml"
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