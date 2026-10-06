#!/usr/bin/env python
"""One self-describing CSV of the whole analysis, plus a column key, for sharing.

    ~/micromamba/envs/bcell-repair/bin/python scripts/13_export_results.py

Writes out/share/af3_vs_crystal_results.csv   - one row per crystal/ASU-copy comparison
      out/share/COLUMNS.md                    - what every column means and its caveats

One row per comparison, not per prediction: a structure with several copies in its
asymmetric unit contributes one row each, because the same prediction scores differently
against them (7NZE: 1.70 A vs 0.86 A) and collapsing that would hide the spread.

Every metric is carried with its own copy-to-copy floor in the same row, so a number can
never be read against the wrong floor. Rows where the prediction and the crystal are not
the same peptide carry caveat=dagger and should not be quoted without the note.
"""
import csv, statistics as st, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SUMMARY = HERE / "out" / "rmsd_summary.tsv"
TOP = HERE / "out" / "peptide_rmsd.tsv"
ALL = HERE / "out" / "peptide_rmsd_all_samples.tsv"
FLOOR = HERE / "out" / "asu_noise_floor.tsv"
TABLE = HERE / "data" / "structures_to_compare.tsv"
OUT = HERE / "out" / "share"

METRICS = [("peptide_ca_rmsd", "median", "ca"),
           ("backbone_rmsd", "median_backbone", "backbone"),
           ("heavy_rmsd", "median_heavy", "heavy"),
           ("core_ca_rmsd", "median_core_ca", "core9")]


def read(p, **kw):
    return list(csv.DictReader((l for l in p.open() if not l.startswith("#")),
                               delimiter="\t", **kw))


def main():
    for p in (SUMMARY, TOP, ALL, FLOOR, TABLE):
        if not p.exists():
            sys.exit(f"{p} missing -- run 05 (--all-samples), 06 and 04 first")
    OUT.mkdir(parents=True, exist_ok=True)

    meta = {r["pdb"]: r for r in read(TABLE)}
    top = {(r["pdb"], r["copy"]): r for r in read(TOP)}
    summ = read(SUMMARY)
    floor_rows = read(FLOOR)
    floors = {}
    for col, _, short in METRICS:
        v = [float(r[col]) for r in floor_rows if r.get(col) not in (None, "")]
        floors[short] = (min(v), max(v)) if v else ("", "")

    spread = defaultdict(lambda: defaultdict(list))
    for r in read(ALL):
        for col, _, short in METRICS:
            if r.get(col):
                spread[(r["pdb"], r["copy"])][short].append(float(r[col]))

    rows = []
    for s in summ:
        k = (s["pdb"], s["copy"])
        m = meta.get(s["pdb"], {})
        t = top.get(k, {})
        row = dict(
            pdb=s["pdb"], asu_copy=s["copy"],
            af3_job=s["job"], predicted_peptide=t.get("pred_seq", ""),
            crystal_peptide_deposited=m.get("entity_seq", ""),
            crystal_peptide_modelled=m.get("modelled_seq", ""),
            prediction_vs_crystal=t.get("pred_vs_deposited", ""),
            caveat=s.get("caveat", ""), modified_residue=m.get("modified_res", ""),
            allele_label=m.get("allele_label", ""),
            n_models=s["n_models"], superposition_frame=s.get("frame", ""),
            mhc_superposition_rmsd=t.get("mhc_align_rmsd", ""),
            n_mhc_ca=t.get("n_mhc_ca", ""),
            n_peptide_residues_scored=t.get("n_pep_ca", ""),
            binding_core_residues=t.get("core_span", ""),
        )
        for col, medcol, short in METRICS:
            v = spread[k].get(short, [])
            row[f"{short}_top_model"] = t.get(col, "")
            row[f"{short}_median"] = s.get(medcol, "")
            # mean, min and max as well as the percentiles: the outlier models are not
            # excluded from anything and must not be invisible in the export either.
            row[f"{short}_mean"] = round(st.mean(v), 3) if v else ""
            row[f"{short}_p5"] = round(st.quantiles(v, n=20)[0], 3) if len(v) > 1 else ""
            row[f"{short}_p95"] = round(st.quantiles(v, n=20)[-1], 3) if len(v) > 1 else ""
            row[f"{short}_min"] = round(min(v), 3) if v else ""
            row[f"{short}_max"] = round(max(v), 3) if v else ""
            row[f"{short}_sd"] = round(st.stdev(v), 3) if len(v) > 1 else ""
            row[f"{short}_variance"] = round(st.variance(v), 3) if len(v) > 1 else ""
            row[f"{short}_crystal_floor_min"] = floors[short][0]
            row[f"{short}_crystal_floor_max"] = floors[short][1]
        row["n_models_over_5A"] = s["n_over_5A"]
        row["seeds_with_failures"] = s["seeds_with_failure"]
        row["note"] = m.get("note", "")
        rows.append(row)

    rows.sort(key=lambda r: (float(r["ca_median"]) if r["ca_median"] else 9e9, r["pdb"]))
    out = OUT / "af3_vs_crystal_results.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    frame = rows[0]["superposition_frame"] if rows else "?"
    (OUT / "COLUMNS.md").write_text(f"""# af3_vs_crystal_results.csv — column key

One row per **crystal / asymmetric-unit copy**, {len(rows)} rows. A PDB entry with
several copies of the complex in its asymmetric unit contributes one row per copy: the
same prediction scores differently against them (7NZE gives 1.70 Å and 0.86 Å), and
collapsing them would hide that spread.

## How the numbers were made

Each AF3 model is superposed onto the crystal **on the MHC only**, then the peptide
RMSD is computed in that frame with **no further superposition** — so a peptide of
correct shape sitting wrong in the groove still scores as wrong. Residues are paired by
sequence and only **identical** residues are compared; a substituted position is
dropped, never mispaired.

`superposition_frame = {frame}`: the fit uses MHC residues within 12 Å of the peptide
(the binding platform) rather than the whole molecule. Class II MHC sits on two Ig-like
domains that hinge, and a whole-MHC fit varies 0.39–5.56 Å across these structures —
8CMF's two copies fit at 0.39 and 3.97 Å for the *same* prediction. That rotation moves
the groove and would be charged to the peptide, which did not move.

## The four metrics

| Prefix | Atoms compared |
|---|---|
| `ca_` | Cα of every paired residue |
| `backbone_` | N, CA, C, O |
| `heavy_` | every non-hydrogen atom the two residues share (names intersected per residue, so citrulline compares against predicted arginine on their common atoms) |
| `core9_` | Cα of the 9-residue binding core |

The binding core is **measured, not assumed**: MHC atoms within 5 Å are counted for
each peptide residue and the contiguous 9-residue window with the most contacts wins.
`binding_core_residues` records which ones — it is often off-centre (8CMF: 4–12).

Each prefix carries:

* `_top_model` — AF3's own top-ranked model
* `_median`, `_mean`, `_p5`, `_p95`, `_min`, `_max` — over all `n_models` models
  (10 seeds × 5 samples). **No model is excluded**; see FAILURES.md for the 46 of 1000
  that put the peptide >5 Å out, what they look like, and how to detect them from AF3's
  own per-chain confidence
* `_crystal_floor_min/max` — **that metric's** copy-to-copy range between independent
  copies of one crystal, measured identically. A value inside this range is
  indistinguishable from the spread the experiment itself shows. **A core number must
  not be read against the whole-peptide floor**, which is why every row carries all four.

## Read these before quoting a row

* `caveat = dagger` — the prediction and the crystal are **not the same peptide**
  (7NZF: two Lys→Ala; 8VCX: an engineered analog, additionally disulfide-bonded to the
  MHC α chain). Only the identical positions are scored. See `note`.
* `prediction_vs_crystal` — `exact`, or how the lengths differ. Flanking residues with
  no counterpart are not scored, so `n_peptide_residues_scored` can be well below the
  predicted length.
* `crystal_peptide_deposited` vs `crystal_peptide_modelled` — the deposited sequence
  versus the residues that actually have coordinates. Disordered termini are missing
  from the latter and cannot be compared against.
* `n_models_over_5A` and `seeds_with_failures` — models that put the peptide right out
  of the groove. They cluster **by seed**, not at random, which is why the median rather
  than the mean is the headline number.
""")
    print(f"{len(rows)} rows -> {out}")
    print(f"column key   -> {OUT / 'COLUMNS.md'}")
    dag = [r["pdb"] for r in rows if r["caveat"] == "dagger"]
    print(f"daggered rows: {', '.join(dag) if dag else 'none'}")


if __name__ == "__main__":
    main()
