#!/usr/bin/env python
"""Build one PDB per comparison holding the crystal and the aligned prediction.

    ~/micromamba/envs/bcell-repair/bin/python scripts/07_prepare_overlays.py

Writes out/overlays/<PDB>_copy<N>.pdb with three chains:

    M = the crystal's MHC groove (residues within GROOVE_R of its peptide)
    P = the crystal peptide          -> drawn TRANSPARENT
    Q = the predicted peptide        -> drawn OPAQUE

Two superpositions happen, in this order:

 1. the prediction is put into ITS crystal's frame on the MHC CAs -- the same operation
    03_peptide_rmsd.py scores, so what the panel shows is exactly what the number
    measures. Nothing is fitted peptide-to-peptide.
 2. every panel is then moved into ONE COMMON frame (VIEW_REF's MHC), applying the same
    transform to crystal and prediction together so their registration is untouched.
    Without this each panel would be viewed down a different axis and the eleven images
    could not be compared side by side.

The RMSD written into out/overlays/labels.tsv is recomputed here from the same code
path, so a panel can never carry a number from a different run.
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
OUT = HERE / "out" / "overlays"

VIEW_REF = "8CMG"      # single-copy DR1; its groove defines the common viewing frame
GROOVE_R = 14.0        # A around the peptide to keep as context
PRED_CHAINS = {"peptide": "PEPTIDE", "mhc_a": "HLA", "mhc_b": "HLB"}


def restrict_frame(ref, rc, rm, pm, frame):
    """Keep only the MHC CAs that 03_peptide_rmsd.py would use for this frame."""
    if frame != "groove":
        return rm, pm
    pep = np.array([a.coord for _, a in R.ca_residues(ref, rc["peptide"])])
    keep = [i for i, a in enumerate(rm)
            if np.linalg.norm(pep - a.coord, axis=1).min() <= R.GROOVE_R]
    return [rm[i] for i in keep], [pm[i] for i in keep]


def ref_chains(pdb, copy):
    out = {}
    for r in csv.DictReader(CHAINS.open(), delimiter="\t"):
        if r["pdb"].upper() == pdb.upper() and int(r["copy"]) == copy:
            out[r["role"]] = r["chain"]
    return out


def mhc_ca(model, chains):
    out = []
    for role in ("mhc_a", "mhc_b"):
        out += R.ca_residues(model, chains[role])
    return out


def view_transform(model, chains, view_atoms):
    """Rigid transform taking this model's MHC onto the common view frame.

    Returned, not applied, because it must be applied to the crystal AND the prediction
    as ONE transform. Fitting them to the view separately changes their relative
    registration -- it moved 1YMM's peptide RMSD from 0.41 to 0.62 A, i.e. it would have
    drawn a panel that disagreed with its own number."""
    a, b = R.pair_by_sequence(mhc_ca(model, chains), view_atoms)
    if len(a) < 50:
        sys.exit(f"only {len(a)} MHC CA pairs against the view reference")
    sup = Superimposer()
    sup.set_atoms(b, a)                      # (fixed=view, moving=this)
    return sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", help="directory of <job>/ prediction folders "
                                        "(default: the one under out/predictions/)")
    args = ap.parse_args()   # not `a`: the loop below rebinds `a`
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader(RMSD.open(), delimiter="\t"))
    frame = next((r.get("frame") for r in csv.DictReader(RMSD.open(), delimiter="\t")
                  if r.get("frame")), "full")
    medians = {}
    if SUMMARY.exists():
        medians = {(r["pdb"], r["copy"]): (r["median"], r["n_over_5A"], r["n_models"])
                   for r in csv.DictReader(SUMMARY.open(), delimiter="\t")}

    # one panel per crystal: the lowest-numbered copy that was scored
    best = {}
    for r in pairs:
        k = r["pdb"]
        if k not in best or int(r["copy"]) < int(best[k]["copy"]):
            best[k] = r
    print(f"{len(best)} panels (one per crystal with a prediction)")

    vmodel = R.load(CIF / f"{VIEW_REF}.cif")
    vchains = ref_chains(VIEW_REF, 1)
    view_atoms = mhc_ca(vmodel, vchains)

    labels = []
    for pdb, row in sorted(best.items()):
        copy = int(row["copy"])
        job = row["job"]
        pred_path = find_model(job, args.pred_root)
        if pred_path is None:
            sys.exit(f"no model.cif for job {job}"
                     + (f" under {args.pred_root}" if args.pred_root else ""))
        rc = ref_chains(pdb, copy)

        ref = R.load(CIF / f"{pdb}.cif")
        pred = R.load(pred_path)

        # (1) prediction -> its crystal, on the MHC. Identical to the scored operation.
        rm, pm = [], []
        for role in ("mhc_a", "mhc_b"):
            a, b = R.pair_by_sequence(R.ca_residues(ref, rc[role]),
                                      R.ca_residues(pred, PRED_CHAINS[role]))
            rm += a; pm += b
        # Use whatever frame the scored table used, or the panels will not reproduce
        # their own numbers -- 07 fitted on the whole MHC while 05 had moved to the
        # groove, and the guard below caught it as 0.408 vs 0.459 on 1YMM.
        rm, pm = restrict_frame(ref, rc, rm, pm, frame)
        sup = Superimposer(); sup.set_atoms(rm, pm); sup.apply(list(pred.get_atoms()))

        rp = R.ca_residues(ref, rc["peptide"])
        pp = R.ca_residues(pred, PRED_CHAINS["peptide"])
        ra, pa = R.pair_by_sequence(rp, pp)
        dev = np.linalg.norm(np.array([x.coord for x in pa]) - np.array([x.coord for x in ra]),
                             axis=1)
        rmsd = float(np.sqrt((dev ** 2).mean()))
        if abs(rmsd - float(row["peptide_ca_rmsd"])) > 0.01:
            sys.exit(f"{pdb}: recomputed {rmsd:.3f} != table {row['peptide_ca_rmsd']}")

        # (2) both into the common view frame
        if pdb != VIEW_REF:
            sup2 = view_transform(ref, rc, view_atoms)
            sup2.apply(list(ref.get_atoms()))
            sup2.apply(list(pred.get_atoms()))      # THE SAME transform, not a refit
            after = np.linalg.norm(np.array([x.coord for x in pa]) -
                                   np.array([x.coord for x in ra]), axis=1)
            moved = float(np.sqrt((after ** 2).mean()))
            if abs(moved - rmsd) > 0.05:
                sys.exit(f"{pdb}: view transform changed the RMSD "
                         f"{rmsd:.3f} -> {moved:.3f}; registration was not preserved")

        # --- assemble the panel ---
        st = Structure.Structure(pdb); mo = Model.Model(0); st.add(mo)
        pep = ref[rc["peptide"]]
        pep_atoms = [a for r_ in pep for a in r_ if r_.id[0] == " " or "CA" in r_]
        pep_xyz = np.array([a.coord for a in pep_atoms])

        cM = Chain.Chain("M")
        for role in ("mhc_a", "mhc_b"):
            for res in ref[rc[role]]:
                if "CA" not in res:
                    continue
                d = np.linalg.norm(pep_xyz - res["CA"].coord, axis=1).min()
                if d <= GROOVE_R:
                    c = res.copy(); c.id = (" ", len(cM) + 1, " "); cM.add(c)
        cP = Chain.Chain("P")
        for i, res in enumerate([r_ for r_ in pep if "CA" in r_], 1):
            c = res.copy(); c.id = (" ", i, " "); cP.add(c)
        cQ = Chain.Chain("Q")
        for i, res in enumerate([r_ for r_ in pred[PRED_CHAINS["peptide"]] if "CA" in r_], 1):
            c = res.copy(); c.id = (" ", i, " "); cQ.add(c)
        for c in (cM, cP, cQ):
            mo.add(c)

        path = OUT / f"{pdb}_copy{copy}.pdb"
        io = PDBIO(); io.set_structure(st); io.save(str(path))
        med, nbad, nmod = medians.get((pdb, str(copy)), ("", "", ""))
        labels.append(dict(pdb=pdb, copy=copy, job=job, pdb_file=path.name,
                           rmsd_top=round(rmsd, 2), rmsd_median=med,
                           n_over_5A=nbad, n_models=nmod,
                           n_groove=len(cM), n_pep_ref=len(cP), n_pep_pred=len(cQ)))
        print(f"  {pdb} copy{copy}: {rmsd:.2f} A  groove={len(cM)} res  "
              f"peptide ref={len(cP)} pred={len(cQ)}")

    with (OUT / "labels.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(labels[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(labels)
    print(f"\n{len(labels)} panels -> {OUT}")


if __name__ == "__main__":
    main()
