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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# Required
JSON_DIR="./data/toy_data/"

# Exactly one of these should be used. If FILE_STEM is non-empty, it wins.
SAMPLE_INDEX=0
FILE_STEM=""

# Main (encoder) graph params
CUTOFF=5.0
MAX_NEIGHBORS=32
GRAPH_METHOD="voronoi"        # radius | voronoi | cov_radius
MIN_FACET_AREA="0.01%"          # e.g. "0.25" or "1.0%"
COV_RADII_SCALE=1.5
SHOW_VORONOI=false

# Attention graph params
ATT_CUTOFF=10.0
ATT_MAX_NEIGHBORS=128
ATT_GRAPH_METHOD="radius"      # radius | voronoi | cov_radius
ATT_MIN_FACET_AREA=""
ATT_COV_RADII_SCALE=1.5

# Drawing / output params
TARGET_SITE_IDX=0
NO_ATOM_LABELS=false
SAVE_PATH=""
NO_SHOW=false

args=(
	"scripts/testing/e3ee_tester.py"
	"--json-dir" "$JSON_DIR"
	"--cutoff" "$CUTOFF"
	"--max-neighbors" "$MAX_NEIGHBORS"
	"--graph-method" "$GRAPH_METHOD"
	"--cov-radii-scale" "$COV_RADII_SCALE"
	"--att-cutoff" "$ATT_CUTOFF"
	"--att-max-neighbors" "$ATT_MAX_NEIGHBORS"
	"--att-graph-method" "$ATT_GRAPH_METHOD"
	"--att-cov-radii-scale" "$ATT_COV_RADII_SCALE"
	"--target-site-idx" "$TARGET_SITE_IDX"
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

if [[ -n "$ATT_MIN_FACET_AREA" ]]; then
	args+=("--att-min-facet-area" "$ATT_MIN_FACET_AREA")
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
