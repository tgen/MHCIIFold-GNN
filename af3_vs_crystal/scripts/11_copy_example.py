#!/usr/bin/env python
"""Worked example: what "copies" are, and how one prediction fits each of them.

    ~/micromamba/envs/bcell-repair/bin/python scripts/11_copy_example.py --pdb 8CME

Writes out/copy_example/<PDB>_asu.pdb        - the whole asymmetric unit, chains tagged
      out/copy_example/<PDB>_copy<N>.pdb     - one overlay panel per copy
      out/copy_example/labels.tsv            - the RMSD that belongs under each panel

Why this exists: "copy" is a crystallographic fact that is invisible in IEDB and in the
RMSD table, and it moves the answer by up to 2x. 8CME deposits THREE copies of the same
DR1/M176-190 complex in one asymmetric unit; a single AF3 prediction scores 1.57, 1.55
and 0.80 A against them. Showing the ASU beside the three overlays makes that concrete.

All panels are put in ONE frame -- copy 1's MHC -- by applying the copy's own
crystal->frame transform to its prediction as well, so the panels are comparable and
each still shows exactly the geometry its number was computed from. (Fitting the two
independently would silently change the RMSD; 07_prepare_overlays.py documents that
trap and this script inherits the same guard.)
"""
import argparse, csv, sys
from pathlib import Path
from importlib import import_module

import numpy as np
from Bio.PDB import Superimposer, PDBIO, Structure, Model, Chain

sys.path.insert(0, str(Path(__file__).resolve().parent))
R = import_module("03_peptide_rmsd")

HERE = Path(__file__).resolve().parent.parent
CIF = HERE / "out" / "crystals"
CHAINS = HERE / "out" / "crystal_chains.tsv"
RMSD = HERE / "out" / "peptide_rmsd.tsv"
SUMMARY = HERE / "out" / "rmsd_summary.tsv"
PRED = HERE / "out" / "predictions"


def find_model(job, pred_root=None):
    """<job>_model.cif, under --pred-root if given else out/predictions/.

    Needed because the predictions do not have to live inside this folder: run_all.sh
    takes PRED_ROOT so the pipeline can be pointed at someone else's AF3 output
    directory. 05 honoured that from the start; this script did not, and a bundle run
    with PRED_ROOT set died here with "no model.cif for job ...".
    """
    if pred_root:
        p = Path(pred_root) / job / f"{job}_model.cif"
        return p if p.exists() else None
    return next(PRED.glob(f"*/{job}/{job}_model.cif"), None)
OUT = HERE / "out" / "copy_example"

GROOVE_R = 14.0
PRED_CHAINS = {"peptide": "PEPTIDE", "mhc_a": "HLA", "mhc_b": "HLB"}


def restrict_frame(ref, rc, rm, pm, frame):
    """Keep only the MHC CAs that 03_peptide_rmsd.py would use for this frame."""
    if frame != "groove":
        return rm, pm
    pep = np.array([a.coord for _, a in R.ca_residues(ref, rc["peptide"])])
    keep = [i for i, a in enumerate(rm)
            if np.linalg.norm(pep - a.coord, axis=1).min() <= R.GROOVE_R]
    return [rm[i] for i in keep], [pm[i] for i in keep]


def chains_of(pdb):
    out = {}
    for r in csv.DictReader(CHAINS.open(), delimiter="\t"):
        if r["pdb"].upper() == pdb.upper():
            out.setdefault(int(r["copy"]), {})[r["role"]] = r["chain"]
    return {k: v for k, v in out.items() if {"peptide", "mhc_a", "mhc_b"} <= set(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="8CME")
    ap.add_argument("--pred-root", help="directory of <job>/ prediction folders "
                                        "(default: the one under out/predictions/)")
    a = ap.parse_args()
    pdb = a.pdb.upper()
    OUT.mkdir(parents=True, exist_ok=True)

    cps = chains_of(pdb)
    if len(cps) < 2:
        sys.exit(f"{pdb} has {len(cps)} complete copy/ies -- pick an entry with several")
    scored = {r["copy"]: r for r in csv.DictReader(RMSD.open(), delimiter="\t")
              if r["pdb"] == pdb}
    frame = next((r.get("frame") for r in csv.DictReader(RMSD.open(), delimiter="\t")
                  if r.get("frame")), "full")
    if not scored:
        sys.exit(f"{pdb} has no scored prediction in {RMSD}")
    job = next(iter(scored.values()))["job"]
    pred_path = find_model(job, a.pred_root)
    if pred_path is None:
        sys.exit(f"no model.cif for job {job}"
                 + (f" under {a.pred_root}" if a.pred_root else ""))
    med = {r["copy"]: r["median"] for r in csv.DictReader(SUMMARY.open(), delimiter="\t")
           if r["pdb"] == pdb} if SUMMARY.exists() else {}
    print(f"{pdb}: {len(cps)} copies in the ASU, prediction = {job}")

    # ---- panel A: the asymmetric unit, every chain relabelled by copy ----
    ref0 = R.load(CIF / f"{pdb}.cif")
    # The PDB format allows ONE character of chain id, so the copy number cannot be
    # written into the chain name. It goes in the B-factor instead, in two DISJOINT
    # ranges so a VMD selection can separate role and copy without arithmetic:
    #     MHC of copy N     -> N        (1, 2, 3, ...)
    #     peptide of copy N -> 10 + N   (11, 12, 13, ...)
    # An earlier version used N and N+0.5, and "beta > 1.4" then also caught copy 2's
    # and copy 3's MHC -- the whole asymmetric unit came out drawn as peptide.
    asu = Structure.Structure(pdb); mo = Model.Model(0); asu.add(mo)
    for cp in sorted(cps):
        for role, ch in cps[cp].items():
            c = Chain.Chain(ch)
            for i, res in enumerate([r_ for r_ in ref0[ch] if "CA" in r_], 1):
                rc = res.copy(); rc.id = (" ", i, " ")
                for at in rc:
                    at.set_bfactor(cp + (10 if role == "peptide" else 0))
                c.add(rc)
            mo.add(c)
    io = PDBIO(); io.set_structure(asu); io.save(str(OUT / f"{pdb}_asu.pdb"))
    print(f"  ASU -> {pdb}_asu.pdb  ({len(list(mo))} chains; B-factor: MHC = copy N, "
          f"peptide = 10+N)")

    # ---- panels B..: one overlay per copy, all in copy 1's frame ----
    view_chains = cps[min(cps)]
    vmodel = R.load(CIF / f"{pdb}.cif")
    view_atoms = []
    for role in ("mhc_a", "mhc_b"):
        view_atoms += R.ca_residues(vmodel, view_chains[role])

    labels = []
    for cp in sorted(cps):
        rc = cps[cp]
        ref = R.load(CIF / f"{pdb}.cif")
        pred = R.load(pred_path)

        rm, pm = [], []
        for role in ("mhc_a", "mhc_b"):
            x, y = R.pair_by_sequence(R.ca_residues(ref, rc[role]),
                                      R.ca_residues(pred, PRED_CHAINS[role]))
            rm += x; pm += y
        # same frame as the scored table, else the panel disagrees with its own label
        rm, pm = restrict_frame(ref, rc, rm, pm, frame)
        sup = Superimposer(); sup.set_atoms(rm, pm); sup.apply(list(pred.get_atoms()))

        ra, pa = R.pair_by_sequence(R.ca_residues(ref, rc["peptide"]),
                                    R.ca_residues(pred, PRED_CHAINS["peptide"]))
        dev = np.linalg.norm(np.array([x.coord for x in pa]) -
                             np.array([x.coord for x in ra]), axis=1)
        rmsd = float(np.sqrt((dev ** 2).mean()))
        want = scored.get(str(cp))
        if want and abs(rmsd - float(want["peptide_ca_rmsd"])) > 0.01:
            sys.exit(f"{pdb} copy{cp}: recomputed {rmsd:.3f} != table "
                     f"{want['peptide_ca_rmsd']}")

        if cp != min(cps):        # bring this copy into copy 1's frame, both together
            ca = []
            for role in ("mhc_a", "mhc_b"):
                ca += R.ca_residues(ref, rc[role])
            x, y = R.pair_by_sequence(ca, view_atoms)
            s2 = Superimposer(); s2.set_atoms(y, x)
            s2.apply(list(ref.get_atoms())); s2.apply(list(pred.get_atoms()))
            after = np.linalg.norm(np.array([q.coord for q in pa]) -
                                   np.array([q.coord for q in ra]), axis=1)
            if abs(float(np.sqrt((after ** 2).mean())) - rmsd) > 0.05:
                sys.exit(f"{pdb} copy{cp}: view transform changed the RMSD")

        st = Structure.Structure(f"{pdb}{cp}"); m2 = Model.Model(0); st.add(m2)
        pep = [r_ for r_ in ref[rc["peptide"]] if "CA" in r_]
        pxyz = np.array([r_["CA"].coord for r_ in pep])
        cM = Chain.Chain("M")
        for role in ("mhc_a", "mhc_b"):
            for res in ref[rc[role]]:
                if "CA" in res and np.linalg.norm(pxyz - res["CA"].coord, axis=1).min() <= GROOVE_R:
                    c = res.copy(); c.id = (" ", len(cM) + 1, " "); cM.add(c)
        cP = Chain.Chain("P")
        for i, res in enumerate(pep, 1):
            c = res.copy(); c.id = (" ", i, " "); cP.add(c)
        cQ = Chain.Chain("Q")
        for i, res in enumerate([r_ for r_ in pred[PRED_CHAINS["peptide"]] if "CA" in r_], 1):
            c = res.copy(); c.id = (" ", i, " "); cQ.add(c)
        for c in (cM, cP, cQ):
            m2.add(c)
        io = PDBIO(); io.set_structure(st); io.save(str(OUT / f"{pdb}_copy{cp}.pdb"))
        labels.append(dict(pdb=pdb, copy=cp, job=job, pdb_file=f"{pdb}_copy{cp}.pdb",
                           rmsd_top=round(rmsd, 2), rmsd_median=med.get(str(cp), ""),
                           chains="/".join(rc[k] for k in ("mhc_a", "mhc_b", "peptide"))))
        print(f"  copy {cp}: chains {labels[-1]['chains']:<10} RMSD {rmsd:.2f} Å")

    with (OUT / "labels.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(labels[0].keys()),
                           delimiter="\t")
        w.writeheader(); w.writerows(labels)
    print(f"\n{len(labels)} copy panels + 1 ASU panel -> {OUT}")


if __name__ == "__main__":
    main()
