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
INPUT="/media/hendrik/ExternalSSD/final_data/tmQM_xas_constgrid/"
OUTPUT="./data/paper_workflow/"

# Optional overrides
TEST_FRACTION=0.2
SPECTRUM_KEY="XANES"
STRATIFY_ABSORBERS="true"
SEED=42

args=(
	"scripts/data_splitting.py"
	"--input" "$INPUT"
	"--output" "$OUTPUT"
	"--test-fraction" "$TEST_FRACTION"
	"--spectrum-key" "$SPECTRUM_KEY"
	"--seed" "$SEED"
)

if [[ "$STRATIFY_ABSORBERS" == "true" ]]; then
	args+=("--stratify-absorbers")
fi

echo "Running: python3 ${args[*]}"
python3 "${args[@]}"