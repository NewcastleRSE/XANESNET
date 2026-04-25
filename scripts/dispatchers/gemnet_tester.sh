#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# gemnet_tester dispatcher
# Edit values below, then run:
#   bash scripts/dispatchers/gemnet_tester.sh
# -----------------------------------------------------------------------------

# Required
JSON_DIR="/media/hendrik/ExternalSSD/final_data/omnixas_test/"

# Exactly one of these should be used. If FILE_STEM is non-empty, it wins.
SAMPLE_INDEX=0
FILE_STEM=""

# Main graph params
CUTOFF=5.0
MAX_NEIGHBORS=50
GRAPH_METHOD="voronoi"       # radius | voronoi | cov_radius
MIN_FACET_AREA="5.0%"           # e.g. "0.25" or "1.0%"
COV_RADII_SCALE=1.5
SHOW_VORONOI=false

# Quadruplet / interaction graph params
ENABLE_QUADRUPLETS=true
INT_CUTOFF=6.0
INT_MAX_NEIGHBORS=50
INT_GRAPH_METHOD="radius"
INT_MIN_FACET_AREA=""
INT_COV_RADII_SCALE=1.5

# GemNet-OC params
ENABLE_OC=true

OC_CUTOFF_AEAINT=5.0
OC_MAX_NEIGHBORS_AEAINT=50
OC_GRAPH_METHOD_AEAINT="voronoi"
OC_MIN_FACET_AREA_AEAINT=""
OC_COV_RADII_SCALE_AEAINT=1.5

OC_CUTOFF_AINT=10.0
OC_MAX_NEIGHBORS_AINT=50
OC_GRAPH_METHOD_AINT="radius"
OC_MIN_FACET_AREA_AINT=""
OC_COV_RADII_SCALE_AINT=1.5

# Drawing / output params
ABSORBER_IDX=0
MAX_TRIPLETS_DRAWN=60
MAX_QUADS_DRAWN=60
MAX_MIXED_DRAWN=60
NO_ATOM_LABELS=false
SAVE_PATH=""
NO_SHOW=false

args=(
	"scripts/gemnet_tester.py"
	"--json-dir" "$JSON_DIR"
	"--cutoff" "$CUTOFF"
	"--max-neighbors" "$MAX_NEIGHBORS"
	"--graph-method" "$GRAPH_METHOD"
	"--cov-radii-scale" "$COV_RADII_SCALE"
	"--int-cutoff" "$INT_CUTOFF"
	"--int-max-neighbors" "$INT_MAX_NEIGHBORS"
	"--int-graph-method" "$INT_GRAPH_METHOD"
	"--int-cov-radii-scale" "$INT_COV_RADII_SCALE"
	"--oc-cutoff-aeaint" "$OC_CUTOFF_AEAINT"
	"--oc-cutoff-aint" "$OC_CUTOFF_AINT"
	"--oc-max-neighbors-aeaint" "$OC_MAX_NEIGHBORS_AEAINT"
	"--oc-max-neighbors-aint" "$OC_MAX_NEIGHBORS_AINT"
	"--oc-graph-method-aeaint" "$OC_GRAPH_METHOD_AEAINT"
	"--oc-cov-radii-scale-aeaint" "$OC_COV_RADII_SCALE_AEAINT"
	"--oc-graph-method-aint" "$OC_GRAPH_METHOD_AINT"
	"--oc-cov-radii-scale-aint" "$OC_COV_RADII_SCALE_AINT"
	"--absorber-idx" "$ABSORBER_IDX"
	"--max-triplets-drawn" "$MAX_TRIPLETS_DRAWN"
	"--max-quads-drawn" "$MAX_QUADS_DRAWN"
	"--max-mixed-drawn" "$MAX_MIXED_DRAWN"
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

if [[ "$ENABLE_QUADRUPLETS" == "true" ]]; then
	args+=("--quadruplets")
fi

if [[ -n "$INT_MIN_FACET_AREA" ]]; then
	args+=("--int-min-facet-area" "$INT_MIN_FACET_AREA")
fi

if [[ "$ENABLE_OC" == "true" ]]; then
	args+=("--oc")
fi

if [[ -n "$OC_MIN_FACET_AREA_AEAINT" ]]; then
	args+=("--oc-min-facet-area-aeaint" "$OC_MIN_FACET_AREA_AEAINT")
fi

if [[ -n "$OC_MIN_FACET_AREA_AINT" ]]; then
	args+=("--oc-min-facet-area-aint" "$OC_MIN_FACET_AREA_AINT")
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
