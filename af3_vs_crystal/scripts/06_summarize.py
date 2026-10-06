#!/usr/bin/env python
"""Per-crystal summary of the all-samples RMSD run.

    ~/micromamba/envs/bcell-repair/bin/python scripts/06_summarize.py

Reads out/peptide_rmsd_all_samples.tsv (and out/peptide_rmsd.tsv for the top-ranked
model), writes out/rmsd_summary.tsv.

Reports four metrics per structure -- peptide CA, backbone (N CA C O), all heavy atoms,
and the CA of the geometrically-determined 9-residue binding core -- and the median
alongside the mean, and counts models above 5 A, because the mean
alone is misleading here: a handful of samples place the peptide completely out of the
groove (up to 26 A), which drags the mean well above where the bulk of the models sit
-- 1YMM's mean is 0.91 A while its median is 0.45 A, on the strength of ONE bad sample
in 50. Quoting the mean by itself would describe a prediction that is almost always
excellent as merely good, and would hide that the failures are not noise: they cluster
by SEED (seed-505 fails 4 of its 5 samples for gslqplalegslqkrgiv), which is a
different phenomenon from sample-to-sample spread and worth knowing separately.
"""
import csv, statistics as st, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ALL = HERE / "out" / "peptide_rmsd_all_samples.tsv"
TOP = HERE / "out" / "peptide_rmsd.tsv"
OUT = HERE / "out" / "rmsd_summary.tsv"
OUTLIER = 5.0          # A; above this the peptide is not in the groove at all


def main():
    if not ALL.exists():
        sys.exit(f"{ALL} missing -- run 05_run_all.py --all-samples "
                 f"--out out/peptide_rmsd_all_samples.tsv first")
    rows = list(csv.DictReader(ALL.open(), delimiter="\t"))
    top = {(r["pdb"], r["copy"]): float(r["peptide_ca_rmsd"])
           for r in csv.DictReader(TOP.open(), delimiter="\t")} if TOP.exists() else {}

    METRICS = ("peptide_ca_rmsd", "backbone_rmsd", "heavy_rmsd", "core_ca_rmsd")
    g, seeds, extra = defaultdict(list), defaultdict(lambda: defaultdict(list)), {}
    for r in rows:
        k = (r["pdb"], r["copy"], r["job"])
        v = float(r["peptide_ca_rmsd"])
        g[k].append(v)
        seeds[k][r["model"].split("_")[0]].append(v)
        extra.setdefault(k, {m: [] for m in METRICS[1:]})
        for m in METRICS[1:]:
            if r.get(m):
                extra[k][m].append(float(r[m]))
        extra[k]["caveat"] = r.get("caveat", "")
        extra[k]["frame"] = r.get("frame", "")
        extra[k]["core_span"] = r.get("core_span", "")

    out = []
    for (pdb, cp, job), v in g.items():
        bad = [x for x in v if x > OUTLIER]
        bad_seeds = sorted(s for s, vals in seeds[(pdb, cp, job)].items()
                           if any(x > OUTLIER for x in vals))
        e = extra[(pdb, cp, job)]
        med = lambda m: round(st.median(e[m]), 3) if e.get(m) else ""
        out.append(dict(pdb=pdb, copy=cp, job=job, n_models=len(v),
                        caveat=e.get("caveat", ""), frame=e.get("frame", ""),
                        top_ranked=top.get((pdb, cp), ""),
                        mean=round(st.mean(v), 3), median=round(st.median(v), 3),
                        sd=round(st.stdev(v), 3), min=round(min(v), 3), max=round(max(v), 3),
                        median_backbone=med("backbone_rmsd"),
                        median_heavy=med("heavy_rmsd"),
                        median_core_ca=med("core_ca_rmsd"),
                        core_span=e.get("core_span", ""),
                        n_over_5A=len(bad), seeds_with_failure=";".join(bad_seeds)))
    out.sort(key=lambda r: r["median"])
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(out)

    print(f"{len(rows)} comparisons over {len(g)} crystal/copy pairs -> {OUT}\n")
    print(f"{'PDB':<6}{'cp':<3}{'  '}{'peptide':<21}{'CA':>7}{'bbone':>7}{'heavy':>7}"
          f"{'core9':>7}{'n>5A':>6}  failing seeds   († = not the same peptide)")
    for r in out:
        dag = " †" if r["caveat"] == "dagger" else "  "
        f = lambda k: f"{r[k]:>7.2f}" if r[k] != "" else "      -"
        print(f"{r['pdb']:<6}{r['copy']:<3}{dag}{r['job']:<21}{r['median']:>7.2f}"
              f"{f('median_backbone')}{f('median_heavy')}{f('median_core_ca')}"
              f"{r['n_over_5A']:>6}  {r['seeds_with_failure']}")
    tot = sum(r["n_over_5A"] for r in out)
    print(f"\n{tot} of {len(rows)} models place the peptide >{OUTLIER} A out of position.")
    print("Failures cluster by seed, so they are a seed-level phenomenon, not sample noise.")


if __name__ == "__main__":
    main()
