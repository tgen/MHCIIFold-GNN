#!/usr/bin/env bash
# Four-panel worked example of what an ASU "copy" is and how much it moves the answer.
#
#   bash scripts/12_render_copy_example.sh [PDB]        # default 8CME
#
# In:  out/copy_example/  (11_copy_example.py)
# Out: out/figures/copy_example_<PDB>.png
#
# Panel 1 is the asymmetric unit with one colour per copy and every peptide in red.
# Panels 2..N are the SAME AF3 prediction (opaque) on each copy's crystal peptide
# (transparent), in that copy's own groove, labelled with the RMSD that copy produced.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDB="${1:-8CME}"
VMD="${VMD:-/Applications/VMD.app/Contents/vmd2/lib/vmd_MACOSXARM64}"
TACHYON="${TACHYON:-/Applications/VMD.app/Contents/vmd2/lib/tachyon_MACOSXARM64}"
export VMDDIR="${VMDDIR:-/Applications/VMD.app/Contents/vmd2/lib}"
FONT="${FONT:-/System/Library/Fonts/Supplemental/Arial.ttf}"
RES="${RES:-1200}"
LABELS="$HERE/out/copy_example/labels.tsv"
[[ -s "$LABELS" ]] || { echo "run 11_copy_example.py first" >&2; exit 1; }
mkdir -p "$HERE/out/panels" "$HERE/out/figures"

render () {   # render <tcl> <in.pdb> <out.png>
    # < /dev/null: VMD reads stdin and would otherwise swallow the caller's loop input
    "$VMD" -dispdev text -e "$1" -args "$2" "/tmp/ce_$$.dat" > "/tmp/ce_$$.log" 2>&1 < /dev/null
    [[ -s "/tmp/ce_$$.dat" ]] || { echo "no scene for $2; see /tmp/ce_$$.log" >&2; exit 1; }
    "$TACHYON" "/tmp/ce_$$.dat" -res "$RES" "$RES" -aasamples 12 -format TARGA \
        -o "/tmp/ce_$$.tga" > /dev/null 2>&1
    magick "/tmp/ce_$$.tga" -trim +repage -bordercolor white -border 4% "$3"
}

asu_png="$HERE/out/panels/${PDB}_asu.png"
render "$HERE/scripts/asu.tcl" "$HERE/out/copy_example/${PDB}_asu.pdb" "$asu_png"
ncopy=$(( $(grep -vc '^$' "$LABELS") - 1 ))
args=(-label "$PDB asymmetric unit: $ncopy copies" "$asu_png")

n=0
while IFS=$'\t' read -r pdb copy job pdb_file rmsd_top rmsd_median chains; do
    [[ "$pdb" == "pdb" ]] && continue
    png="$HERE/out/panels/${pdb}_ce_copy${copy}.png"
    render "$HERE/scripts/overlay.tcl" "$HERE/out/copy_example/$pdb_file" "$png"
    args+=(-label "copy $copy  (chains $chains)   ${rmsd_top} A" "$png")
    n=$((n+1))
done < "$LABELS"

(( n == ncopy )) || { echo "rendered $n of $ncopy copies -- refusing to montage" >&2; exit 1; }
montage -font "$FONT" -pointsize 30 -background white -fill black \
    "${args[@]}" -tile $((n + 1))x1 -geometry +10+10 \
    "$HERE/out/figures/copy_example_${PDB}.png"
echo "$((n + 1)) panels -> $HERE/out/figures/copy_example_${PDB}.png"
