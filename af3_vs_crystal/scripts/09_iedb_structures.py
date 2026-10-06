#!/usr/bin/env python
"""Ask IEDB which of the source epitopes it links to a PDB structure.

    ~/micromamba/envs/bcell-repair/bin/python scripts/09_iedb_structures.py

Writes out/iedb_structures.tsv and prints how IEDB's answer compares to the RCSB
sequence scan that built data/structures_to_compare.tsv.

Kameron's epitope table comes from IEDB, and IEDB records a `pdb_ids` field per
epitope, so this is the list's own native answer to "which of these have structures" --
a different route to the same question than searching RCSB by sequence. The two
disagree in both directions and the disagreement is the point:

  * IEDB lags for recent depositions. It links GIAGFKGEQGPKGEP to 2FSE only, while
    RCSB also has 6NIX (2020) and 7NZE (2022) with that exact peptide.
  * IEDB knows things the CSV does not. GQVELGGGPGAESCQ is an IEDB epitope linked to
    8VCX and 8VDD, but it is absent from epitopeHLAPositive.csv -- the CSV is a
    filtered subset, so a structure can be missing from the CSV yet present upstream.

Queried through the public query API rather than a local database copy:
data/iedb_public.db is a truncated 0.68% fragment of the cluster file and every query
against it fails (docs/PROVENANCE.md records this).
"""
import csv, json, re, sys, time, urllib.parse, urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CSV_IN = HERE / "data" / "epitopeHLAPositive.csv"
TABLE = HERE / "data" / "structures_to_compare.tsv"
OUT = HERE / "out" / "iedb_structures.tsv"

API = "https://query-api.iedb.org/epitope_search"
BATCH = 40


def fetch(seqs, tries=4):
    q = urllib.parse.urlencode({
        "linear_sequence": "in.(" + ",".join(seqs) + ")",
        "select": "structure_id,linear_sequence,pdb_ids,mhc_allele_names,"
                  "parent_source_antigen_names",
        "limit": "500"})
    for a in range(tries):
        try:
            req = urllib.request.Request(f"{API}?{q}",
                                         headers={"User-Agent": "tcr-af3-triad/iedb-check"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if a == tries - 1:
                sys.exit(f"IEDB query failed after {tries} tries: {e!r}")
            time.sleep(2 * (a + 1))


def main():
    eps = sorted({re.sub(r"[^A-Z]", "", r["Original"].upper())
                  for r in csv.DictReader(CSV_IN.open())})
    print(f"{len(eps)} unique epitopes -> IEDB, {BATCH} per call", flush=True)

    rows, seen = [], set()
    for i in range(0, len(eps), BATCH):
        for r in fetch(eps[i:i + BATCH]):
            pdbs = r.get("pdb_ids") or []
            key = (r["linear_sequence"], r["structure_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(epitope=r["linear_sequence"], structure_id=r["structure_id"],
                             pdb_ids=";".join(sorted(p.upper() for p in pdbs)),
                             antigen=";".join((r.get("parent_source_antigen_names") or [])[:2]),
                             alleles=";".join(sorted(set(r.get("mhc_allele_names") or []))[:6])))
        if (i // BATCH) % 4 == 0:
            print(f"  {min(i + BATCH, len(eps))}/{len(eps)}", flush=True)
        time.sleep(0.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["epitope"]))

    with_pdb = defaultdict(set)
    for r in rows:
        for p in filter(None, r["pdb_ids"].split(";")):
            with_pdb[r["epitope"]].add(p)
    iedb_pdbs = {p for v in with_pdb.values() for p in v}
    ours = {r["pdb"] for r in csv.DictReader(
        (l for l in TABLE.open() if not l.startswith("#")), delimiter="\t")}

    print(f"\n{len(rows)} IEDB epitope records for {len({r['epitope'] for r in rows})} "
          f"of {len(eps)} epitopes -> {OUT}")
    print(f"epitopes IEDB links to a PDB entry : {len(with_pdb)}")
    print(f"distinct PDB entries IEDB names    : {len(iedb_pdbs)}")
    print(f"\nin BOTH IEDB and our set ({len(iedb_pdbs & ours)}): {' '.join(sorted(iedb_pdbs & ours))}")
    print(f"\nIEDB names, we do NOT have ({len(iedb_pdbs - ours)}):")
    for p in sorted(iedb_pdbs - ours):
        who = [e for e, v in with_pdb.items() if p in v]
        print(f"    {p}  {who[0] if who else ''}")
    print(f"\nwe have, IEDB does not name ({len(ours - iedb_pdbs)}): {' '.join(sorted(ours - iedb_pdbs))}")
    print("\nNeither list is a superset. IEDB lags on recent depositions; the RCSB scan "
          "misses\nepitopes the CSV filtered out. Use the union, and check anything that "
          "appears in only one.")


if __name__ == "__main__":
    main()
