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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# Required
JSON_DIR="./data/toy_data/"

# Exactly one of these should be used. If FILE_STEM is non-empty, it wins.
SAMPLE_INDEX="${SAMPLE_INDEX:-2}"
# 10
FILE_STEM=""

# Graph parameters
CUTOFF=3.0
MAX_NEIGHBORS=50
GRAPH_METHOD="cov_radius"       # radius | voronoi | cov_radius
MIN_FACET_AREA="1.0%"           # e.g. "0.25" or "1.0%" (voronoi only)
COV_RADII_SCALE=1.5         # cov_radius only

# Optional toggles
SHOW_VORONOI=false
NO_ATOM_LABELS=false
NO_SHOW=false

# Target-site / sampling
TARGET_SITE_IDX=0
MAX_TRIPLETS_DRAWN=60
MAX_PATHS_DRAWN=60
MAX_PATHS=128

# Optional output path (empty = do not save)
SAVE_PATH="./runs/testing/graph_tester.png"

args=(
	"scripts/testing/graph_tester.py"
	"--json-dir" "$JSON_DIR"
	"--cutoff" "$CUTOFF"
	"--max-neighbors" "$MAX_NEIGHBORS"
	"--graph-method" "$GRAPH_METHOD"
	"--cov-radii-scale" "$COV_RADII_SCALE"
	"--target-site-idx" "$TARGET_SITE_IDX"
	"--max-triplets-drawn" "$MAX_TRIPLETS_DRAWN"
	"--max-paths-drawn" "$MAX_PATHS_DRAWN"
	"--max-paths" "$MAX_PATHS"
)

if [[ -n "$FILE_STEM" ]]; then
	args+=("--file" "$FILE_STEM")
else
	args+=("--index" "$SAMPLE_INDEX")
fi

if [[ -n "$MIN_FACET_AREA" ]]; then
	args+=("--min-facet-area" "$MIN_FACET_AREA")
fi

if [[ "$SHOW_VORONOI" == "true" ]]; then
	args+=("--show-voronoi")
fi

if [[ "$NO_ATOM_LABELS" == "true" ]]; then
	args+=("--no-atom-labels")
fi

if [[ -n "$SAVE_PATH" ]]; then
	args+=("--save" "$SAVE_PATH")
fi

if [[ "$NO_SHOW" == "true" ]]; then
	args+=("--no-show")
fi

echo "Running: python3 ${args[*]}"
python3 "${args[@]}"
