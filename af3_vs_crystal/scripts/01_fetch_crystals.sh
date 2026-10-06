#!/usr/bin/env bash
# Download the crystal structures named in data/structures_to_compare.tsv from RCSB.
# Only rows marked status=include are fetched; EXCLUDE rows are reported by name so a
# deliberate exclusion never looks like a download that quietly went missing.
#
#   bash scripts/01_fetch_crystals.sh            # -> out/crystals/<PDB>.cif
#
# Re-running is cheap: an entry already on disk is skipped. Every requested ID must
# land in exactly one bucket (fetched / already present / FAILED) and the buckets are
# reconciled against the input count at the end -- a partial download that reports
# success is the failure mode this repo keeps hitting.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIST="$HERE/data/structures_to_compare.tsv"
DEST="$HERE/out/crystals"
mkdir -p "$DEST"

# bash 3.2 (macOS default) has no mapfile
# Columns are found BY HEADER NAME, never by position: this file has gained columns
# twice, and a hardcoded $5 silently pointed at the wrong field, marking all 23 entries
# excluded and requesting none.
# tr -d '\r' because these TSVs have been written with CRLF endings: python's
# csv.DictWriter defaults to \r\n, python readers hide it, and awk does not -- the
# last column silently gains a carriage return.
col () {   # col <header-name> -> 1-based field index
    grep -v '^#' "$LIST" | head -1 | tr -d '\r' | tr '\t' '\n' | grep -nx "$1" | cut -d: -f1
}
C_PDB=$(col pdb); C_STATUS=$(col status); C_NOTE=$(col note)
for v in C_PDB C_STATUS C_NOTE; do
    [[ -n "${!v}" ]] || { echo "column for $v not found in $LIST header" >&2; exit 2; }
done

IDS=()
while IFS= read -r id; do IDS+=("$id"); done < <(
    grep -v '^#' "$LIST" | tail -n +2 | tr -d '\r' |
    awk -F'\t' -v p="$C_PDB" -v s="$C_STATUS" '$s=="include"{print $p}' |
    grep -v '^$' | sort -u)
EXCL=()
while IFS= read -r line; do EXCL+=("$line"); done < <(
    grep -v '^#' "$LIST" | tail -n +2 | tr -d '\r' |
    awk -F'\t' -v p="$C_PDB" -v s="$C_STATUS" -v n="$C_NOTE" \
        '$s!="include"{print $p" ("$n")"}')

if (( ${#IDS[@]} == 0 )); then
    echo "no rows with status=include in $LIST -- refusing to continue" >&2
    exit 2
fi
echo "requested: ${#IDS[@]} distinct PDB IDs"
if (( ${#EXCL[@]} )); then
    echo "deliberately excluded (${#EXCL[@]}):"
    printf '    %s\n' "${EXCL[@]}"
fi

fetched=0; present=0; failed=()
for id in "${IDS[@]}"; do
    out="$DEST/${id}.cif"
    if [[ -s "$out" ]]; then present=$((present+1)); continue; fi
    if curl -sfL "https://files.rcsb.org/download/${id}.cif" -o "$out.tmp" && [[ -s "$out.tmp" ]]; then
        mv "$out.tmp" "$out"; fetched=$((fetched+1))
    else
        rm -f "$out.tmp"; failed+=("$id")
    fi
done

echo "fetched=$fetched already_present=$present failed=${#failed[@]}"
if (( fetched + present + ${#failed[@]} != ${#IDS[@]} )); then
    echo "BUG: buckets do not sum to the request count" >&2; exit 2
fi
if (( ${#failed[@]} )); then
    printf 'FAILED: %s\n' "${failed[@]}" >&2
    echo "refusing to report success with missing structures" >&2; exit 1
fi
echo "all ${#IDS[@]} structures in $DEST"
