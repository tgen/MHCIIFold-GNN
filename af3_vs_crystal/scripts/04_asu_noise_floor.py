#!/usr/bin/env python
"""Peptide RMSD between independent copies of the SAME complex in one crystal.

    ~/micromamba/envs/bcell-repair/bin/python scripts/04_asu_noise_floor.py

Writes out/asu_noise_floor.tsv.

Three of the fetched entries hold more than one copy of the complex in the asymmetric
unit (6HBY x2, 8CMF x2, 8CME x3). Those copies are the same molecule, solved in the
same experiment, at the same resolution — so the peptide RMSD between them is a floor
on what any prediction can be asked to beat. An AF3 peptide RMSD at or below this
range is not "wrong by X angstroms"; it is inside the spread the crystal itself shows.

Measured exactly as 03_peptide_rmsd.py measures a prediction: superpose on the MHC of
the two copies, then peptide CA RMSD with no further superposition. Calibrating the
reference the same way the samples are measured is the point — a floor computed a
different way would not be comparable to the numbers it is meant to bound.
"""
import csv, itertools, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
rmsd_mod = import_module("03_peptide_rmsd")

HERE = Path(__file__).resolve().parent.parent
CHAINS = HERE / "out" / "crystal_chains.tsv"
CIF = HERE / "out" / "crystals"
OUT = HERE / "out" / "asu_noise_floor.tsv"
# Must match the frame 05_run_all.py scores in, or the floor cannot bound those numbers.
FRAME = "groove"


def copies_of(pdb):
    """copy number -> {role: chain} for every complete copy in the ASU."""
    out = {}
    with CHAINS.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["pdb"].upper() != pdb.upper():
                continue
            out.setdefault(int(r["copy"]), {})[r["role"]] = r["chain"]
    return {k: v for k, v in out.items()
            if {"peptide", "mhc_a", "mhc_b"} <= set(v)}


def main():
    if not CHAINS.exists():
        sys.exit(f"{CHAINS} missing -- run 02_describe_crystals.py first")
    pdbs = sorted({r["pdb"] for r in csv.DictReader(CHAINS.open(), delimiter="\t")})

    rows, skipped = [], []
    for pdb in pdbs:
        cps = copies_of(pdb)
        if len(cps) < 2:
            skipped.append((pdb, len(cps)))
            continue
        path = CIF / f"{pdb}.cif"
        for i, j in itertools.combinations(sorted(cps), 2):
            r = rmsd_mod.peptide_rmsd(path, path, cps[i], cps[j], frame=FRAME,
                                      label=f"{pdb} copy{i} vs copy{j}")
            rnd = lambda v: round(v, 3) if v is not None else ""
            rows.append(dict(pdb=pdb, copy_a=i, copy_b=j, frame=FRAME,
                             peptide_ca_rmsd=round(r["peptide_rmsd"], 3),
                             backbone_rmsd=rnd(r["backbone_rmsd"]),
                             heavy_rmsd=rnd(r["heavy_rmsd"]),
                             core_ca_rmsd=rnd(r["core_ca_rmsd"]),
                             core_span=r["core_span"],
                             n_pep_ca=r["n_pep_ca"],
                             mhc_align_rmsd=round(r["mhc_align_rmsd"], 3),
                             n_mhc_ca=r["n_mhc_ca"],
                             worst_residue=round(r["max_dev"], 3)))

    if not rows:
        sys.exit("no multi-copy entries found -- nothing to calibrate against")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} copy-pairs from {len({r['pdb'] for r in rows})} entries "
          f"[{FRAME} frame] -> {OUT}\n")
    print(f"{'entry':<7}{'pair':<17}{'CA':>7}{'bbone':>7}{'heavy':>7}{'core9':>7}{'worst':>8}")
    for r in rows:
        f = lambda k: f"{r[k]:>7.2f}" if r[k] != "" else "      -"
        print(f"{r['pdb']:<7}copy{r['copy_a']} vs copy{r['copy_b']:<9}"
              f"{r['peptide_ca_rmsd']:>7.2f}{f('backbone_rmsd')}{f('heavy_rmsd')}"
              f"{f('core_ca_rmsd')}{r['worst_residue']:>8.2f}")
    print()
    for key, name in (("peptide_ca_rmsd", "peptide CA"), ("backbone_rmsd", "backbone"),
                      ("heavy_rmsd", "heavy atom"), ("core_ca_rmsd", "9-mer core CA")):
        v = [r[key] for r in rows if r[key] != ""]
        if v:
            print(f"  noise floor, {name:<14}: {min(v):.2f}-{max(v):.2f} A")
    print("\nEach metric gets its OWN floor, measured the same way it measures a\n"
          "prediction. A core-CA number cannot be judged against a full-peptide CA floor.")
    if skipped:
        print(f"\nsingle-copy entries, no floor available ({len(skipped)}): "
              + ", ".join(p for p, _ in skipped))


if __name__ == "__main__":
    main()
