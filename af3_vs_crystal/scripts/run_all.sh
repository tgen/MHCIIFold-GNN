#!/usr/bin/env bash
# Whole pipeline, in order, from an empty out/ to both figures.
#
#   bash scripts/run_all.sh                          # top-ranked model only (~2 min)
#   ALL_SAMPLES=1 bash scripts/run_all.sh             # all 50 models each    (~6 min)
#   PRED_ROOT=/path/to/fullLengthAlphaPDB bash scripts/run_all.sh
#
# PRED_ROOT points at the directory that CONTAINS the per-job folders (one folder per
# job, each named after the lowercased peptide and holding <job>_model.cif plus
# seed-*_sample-*/model.cif). Without it, the single directory under out/predictions/
# is used.
#
# Prerequisite that this script cannot supply: the AF3 predictions. They are Kameron's
# (Altin lab) and live in out/predictions/<root>/<job>/, one directory per job, each
# job named after the lowercased peptide. Nothing else needs fetching by hand.
#
# Each step is skipped if its own inputs are missing, with a message -- so a partial
# checkout gives a clear "this step needs X" rather than a stack trace. The steps are
# ordered by dependency, not by preference; 06 needs 05 --all-samples, 07 needs 05 and
# 06, 08 needs 07, 10 needs 04 and 06.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/micromamba/envs/bcell-repair/bin/python}"
cd "$HERE"

[[ -x "$PY" ]] || { echo "interpreter not found: $PY" >&2
                    echo "  (repo analysis env; Homebrew python3 has no numpy)" >&2; exit 1; }

step () { printf '\n=== %s\n' "$*"; }

step "01  fetch crystal structures from RCSB"
bash scripts/01_fetch_crystals.sh

step "02  inventory chains and assign roles"
"$PY" scripts/02_describe_crystals.py

step "03  (library) peptide RMSD -- imported by 04/05/07, not run directly"

step "04  crystallographic noise floor from multi-copy entries"
"$PY" scripts/04_asu_noise_floor.py

PRED_ARG=()
if [[ -n "${PRED_ROOT:-}" ]]; then
    [[ -d "$PRED_ROOT" ]] || { echo "PRED_ROOT is not a directory: $PRED_ROOT" >&2; exit 1; }
    PRED_ARG=(--pred-root "$PRED_ROOT")
    echo "using predictions from $PRED_ROOT"
elif [[ ! -d out/predictions ]] || [[ -z "$(ls -A out/predictions 2>/dev/null)" ]]; then
    echo
    echo "out/predictions/ is empty -- steps 05-08 and 10 need the AF3 predictions."
    echo "Put them at out/predictions/<root>/<job>/<job>_model.cif and re-run."
    exit 0
fi

step "05  score every prediction against every matching crystal"
"$PY" scripts/05_run_all.py "${PRED_ARG[@]}"
if [[ "${ALL_SAMPLES:-0}" == "1" ]]; then
    step "05b all 50 seed/sample models"
    "$PY" scripts/05_run_all.py "${PRED_ARG[@]}" --all-samples \
        --out out/peptide_rmsd_all_samples.tsv
fi

if [[ -s out/peptide_rmsd_all_samples.tsv ]]; then
    step "06  per-structure summary (median, spread, failures by seed)"
    "$PY" scripts/06_summarize.py
else
    echo "(skipping 06: needs out/peptide_rmsd_all_samples.tsv -- run with ALL_SAMPLES=1)"
fi

step "07  build overlay PDBs in a common MHC frame"
"$PY" scripts/07_prepare_overlays.py "${PRED_ARG[@]}"

step "08  render overlay panels + montage  (needs VMD and ImageMagick)"
if command -v magick > /dev/null && [[ -x "${VMD:-/Applications/VMD.app/Contents/vmd2/lib/vmd_MACOSXARM64}" ]]; then
    bash scripts/08_render_overlays.sh
else
    echo "(skipping 08: VMD or ImageMagick not found -- see the script header for paths)"
fi

if [[ -s out/rmsd_summary.tsv ]]; then
    step "10  dot plot"
    "$PY" scripts/10_plot_rmsd.py
    step "13  shareable CSV + column key"
    "$PY" scripts/13_export_results.py
else
    echo "(skipping 10/13: need out/rmsd_summary.tsv from 06 -- run with ALL_SAMPLES=1)"
fi

echo
echo "09_iedb_structures.py is a cross-check against IEDB, not part of the measurement."
echo "It hits the network, so run it on its own when you want it."
printf '\ndone. figures in out/figures/\n'
