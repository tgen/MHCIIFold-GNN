#!/usr/bin/env bash
# Render every overlay panel and montage them into one labelled figure.
#
#   bash scripts/08_render_overlays.sh
#
# In:  out/overlays/*.pdb + out/overlays/labels.tsv   (07_prepare_overlays.py)
# Out: out/panels/<PDB>.png  and  out/figures/overlay_montage.png
#
# Blue + transparent = crystal peptide. Red + opaque = AF3 prediction. Grey ghost =
# the crystal's MHC groove. All panels are already in a COMMON MHC frame, so the same
# camera is used for every one and the images are directly comparable -- rotating them
# individually would make the comparison meaningless.
#
# Note when reading the panels: the whole predicted peptide is drawn, including
# residues with no counterpart in the crystal (4Y19/4Y1A predict 18 residues where
# only 13-14 are modelled). The RMSD in each label covers ONLY the paired residues,
# so an overhanging red tail is not part of the number.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VMD="${VMD:-/Applications/VMD.app/Contents/vmd2/lib/vmd_MACOSXARM64}"
TACHYON="${TACHYON:-/Applications/VMD.app/Contents/vmd2/lib/tachyon_MACOSXARM64}"
export VMDDIR="${VMDDIR:-/Applications/VMD.app/Contents/vmd2/lib}"
FONT="${FONT:-/System/Library/Fonts/Supplemental/Arial.ttf}"
RES="${RES:-1100}"

for exe in "$VMD" "$TACHYON"; do
    [[ -x "$exe" ]] || { echo "not executable: $exe" >&2; exit 1; }
done
[[ -f "$FONT" ]] || { echo "font not found: $FONT" >&2; exit 1; }
# VMDDIR must hold scripts/vmd/vmdinit.tcl or VMD renders with its Tcl init silently
# missing -- the failure is invisible in the output, so check it here.
[[ -f "$VMDDIR/scripts/vmd/vmdinit.tcl" ]] || {
    echo "VMDDIR=$VMDDIR does not contain scripts/vmd/vmdinit.tcl" >&2; exit 1; }

LABELS="$HERE/out/overlays/labels.tsv"
[[ -s "$LABELS" ]] || { echo "run 07_prepare_overlays.py first" >&2; exit 1; }
mkdir -p "$HERE/out/panels" "$HERE/out/figures"

n=0; args=()
while IFS=$'\t' read -r pdb copy job pdb_file rmsd_top rmsd_median n_over n_models rest; do
    [[ "$pdb" == "pdb" ]] && continue
    src="$HERE/out/overlays/$pdb_file"
    [[ -s "$src" ]] || { echo "missing $src" >&2; exit 1; }
    png="$HERE/out/panels/${pdb}.png"
    # < /dev/null is load-bearing: VMD reads stdin, and without this it swallows the
    # rest of labels.tsv, so the loop renders ONE panel and still exits 0.
    "$VMD" -dispdev text -e "$HERE/scripts/overlay.tcl" -args "$src" "/tmp/${pdb}.dat" \
        > "/tmp/vmd_${pdb}.log" 2>&1 < /dev/null
    [[ -s "/tmp/${pdb}.dat" ]] || { echo "no scene for $pdb; see /tmp/vmd_${pdb}.log" >&2; exit 1; }
    "$TACHYON" "/tmp/${pdb}.dat" -res "$RES" "$RES" -aasamples 12 -format TARGA \
        -o "/tmp/${pdb}.tga" > /dev/null 2>&1
    magick "/tmp/${pdb}.tga" -trim +repage -bordercolor white -border 3% "$png"
    lab="${pdb}   ${rmsd_top} A"
    [[ -n "${rmsd_median:-}" ]] && lab="${lab}  (median ${rmsd_median}, ${n_over}/${n_models} >5A)"
    args+=(-label "$lab" "$png")
    n=$((n+1))
    echo "  rendered $pdb  ${rmsd_top} A"
done < "$LABELS"

expected=$(( $(grep -vc '^$' "$LABELS") - 1 ))     # minus the header
if (( n != expected )); then
    echo "rendered $n panels but labels.tsv lists $expected -- refusing to montage a subset" >&2
    exit 1
fi
montage -font "$FONT" -pointsize 30 -background white -fill black \
    "${args[@]}" -tile 4x -geometry +10+10 "$HERE/out/figures/overlay_montage.png"
echo "$n panels -> $HERE/out/figures/overlay_montage.png"
