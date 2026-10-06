#!/usr/bin/env python
"""Peptide RMSD for every prediction against every crystal it corresponds to.

    ~/micromamba/envs/bcell-repair/bin/python scripts/05_run_all.py            # top model
    ~/micromamba/envs/bcell-repair/bin/python scripts/05_run_all.py --all-samples

Writes out/peptide_rmsd.tsv.

Kameron's AF3 jobs are named after the lowercased peptide sequence
(`enpvvhffknivtpr/`), so the job name IS the predicted peptide and no separate mapping
table is needed. Chains in his output are named HLA / HLB / PEPTIDE. The MHC is
FULL LENGTH, including the signal peptide the crystal does not have, so the MHC
superposition pairs only the residues the two actually share.

A job is matched to a crystal when the predicted sequence equals that crystal's
deposited peptide (`entity_seq`), or one contains the other. Every job and every
crystal is then reported in exactly one bucket -- matched, prediction-without-crystal,
or crystal-without-prediction -- and the buckets are reconciled against the totals. A
crystal silently left out is the failure this repo keeps hitting.
"""
import argparse, csv, sys
from pathlib import Path
from importlib import import_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
rmsd_mod = import_module("03_peptide_rmsd")

HERE = Path(__file__).resolve().parent.parent
PRED = HERE / "out" / "predictions"
CIF = HERE / "out" / "crystals"
TABLE = HERE / "data" / "structures_to_compare.tsv"
CHAINS = HERE / "out" / "crystal_chains.tsv"
OUT = HERE / "out" / "peptide_rmsd.tsv"

PRED_CHAINS = {"peptide": "PEPTIDE", "mhc_a": "HLA", "mhc_b": "HLB"}


def crystals():
    rows = [r for r in csv.DictReader(
        (l for l in TABLE.open() if not l.startswith("#")), delimiter="\t")
        if r["status"] == "include"]
    return rows


def copies_of(pdb):
    out = {}
    for r in csv.DictReader(CHAINS.open(), delimiter="\t"):
        if r["pdb"].upper() == pdb.upper():
            out.setdefault(int(r["copy"]), {})[r["role"]] = r["chain"]
    return {k: v for k, v in out.items() if {"peptide", "mhc_a", "mhc_b"} <= set(v)}


def jobs(pred_root=None):
    if pred_root:
        root = Path(pred_root)
        if not root.is_dir():
            sys.exit(f"--pred-root {root} is not a directory")
    else:
        root = next((d for d in PRED.iterdir() if d.is_dir()), None)
        if root is None:
            sys.exit(f"no prediction directory under {PRED}")
    out = {}
    for d in sorted(root.iterdir()):
        top = d / f"{d.name}_model.cif"
        if d.is_dir() and top.exists():
            out[d.name.upper()] = d
    if not out:
        sys.exit(f"no <job>/<job>_model.cif found under {root}")
    return out


def relation(pred, dep, caveat=""):
    """How the folded peptide relates to the deposited one, or None for "no match".

    A row marked caveat=dagger in structures_to_compare.tsv is matched even when the
    sequences differ by substitutions (7NZF, 8VCX): those comparisons were asked for
    explicitly, are scored over the identical positions only, and carry the dagger
    everywhere they appear. Without the marker they would not match at all.
    """
    if pred == dep:
        return "exact"
    if pred in dep:
        return f"pred_shorter_by_{len(dep) - len(pred)}"
    if dep in pred:
        return f"pred_longer_by_{len(pred) - len(dep)}"
    if caveat == "dagger" and len(pred) and len(dep):
        s, l = (pred, dep) if len(pred) <= len(dep) else (dep, pred)
        best = min(sum(1 for x, y in zip(s, l[i:i + len(s)]) if x != y)
                   for i in range(len(l) - len(s) + 1))
        if best <= 3:
            return f"substituted_{best}"
    return None


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-samples", action="store_true",
                    help="score all 50 seed/sample models, not just the top-ranked one")
    ap.add_argument("--pred-root",
                    help="directory of <job>/ prediction folders. Default: the single "
                         "directory under out/predictions/. On Gemini, point this at "
                         "Kameron's originals rather than copying them.")
    ap.add_argument("--frame", choices=("full", "groove"), default="groove",
                    help="MHC superposition frame: the binding platform within 12 A of "
                         "the peptide (default) or the whole MHC. See 03's docstring -- "
                         "the whole-MHC fit varies 0.39-5.56 A across this set because "
                         "the Ig-like domains hinge, and that rotation is charged to the "
                         "peptide.")
    ap.add_argument("--out", help=f"output TSV (default {OUT})")
    a = ap.parse_args()

    if a.out:
        OUT = Path(a.out)
    xtals, J = crystals(), jobs(a.pred_root)
    print(f"{len(J)} prediction jobs, {len(xtals)} crystals in the comparison set\n")

    pairs, unmatched_pred = [], []
    for seq, d in J.items():
        hits = [x for x in xtals
                if relation(seq, x["entity_seq"], x.get("caveat", ""))]
        if not hits:
            unmatched_pred.append((seq, d.name))
            continue
        for x in hits:
            pairs.append((seq, d, x, relation(seq, x["entity_seq"], x.get("caveat", ""))))
    matched_xtal = {x["pdb"] for _, _, x, _ in pairs}
    unmatched_xtal = [x for x in xtals if x["pdb"] not in matched_xtal]

    rows = []
    for seq, d, x, rel in pairs:
        models = [("top", d / f"{d.name}_model.cif")]
        if a.all_samples:
            models = sorted((f"{p.parent.name}", p) for p in d.glob("seed-*_sample-*/model.cif"))
        cps = copies_of(x["pdb"])
        for copy, ch in sorted(cps.items()):
            for tag, path in models:
                try:
                    r = rmsd_mod.peptide_rmsd(CIF / f"{x['pdb']}.cif", path, ch,
                                              PRED_CHAINS, frame=a.frame)
                except SystemExit as e:
                    print(f"  !! {x['pdb']}#{copy} vs {d.name} [{tag}]: {e}", file=sys.stderr)
                    continue
                rnd = lambda v: round(v, 3) if v is not None else ""
                rows.append(dict(pdb=x["pdb"], copy=copy, job=d.name, model=tag,
                                 pred_seq=seq, entity_seq=x["entity_seq"],
                                 pred_vs_deposited=rel, caveat=x.get("caveat", ""),
                                 frame=a.frame,
                                 peptide_ca_rmsd=rnd(r["peptide_rmsd"]),
                                 backbone_rmsd=rnd(r["backbone_rmsd"]),
                                 heavy_rmsd=rnd(r["heavy_rmsd"]),
                                 core_ca_rmsd=rnd(r["core_ca_rmsd"]),
                                 core_span=r["core_span"],
                                 n_pep_ca=r["n_pep_ca"], n_backbone=r["n_backbone"],
                                 n_heavy=r["n_heavy"], n_core=r["n_core"],
                                 worst_residue=round(r["max_dev"], 3),
                                 mhc_align_rmsd=round(r["mhc_align_rmsd"], 3),
                                 n_mhc_ca=r["n_mhc_ca"],
                                 source=x["source"], modified_res=x["modified_res"]))

    if not rows:
        sys.exit("no comparisons produced")
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} comparisons -> {OUT}\n")
    print(f"\n{'PDB':<6}{'cp':<3}{'  '}{'job':<22}{'CA':>7}{'bbone':>7}{'heavy':>7}"
          f"{'core9':>7}   pred vs deposited     († = not the same peptide)")
    for r in sorted(rows, key=lambda r: r["peptide_ca_rmsd"]):
        if r["model"] != "top":
            continue
        dag = " †" if r["caveat"] == "dagger" else "  "
        bb = f"{r['backbone_rmsd']:>7.2f}" if r["backbone_rmsd"] != "" else "      -"
        hv = f"{r['heavy_rmsd']:>7.2f}" if r["heavy_rmsd"] != "" else "      -"
        co = f"{r['core_ca_rmsd']:>7.2f}" if r["core_ca_rmsd"] != "" else "      -"
        print(f"{r['pdb']:<6}{r['copy']:<3}{dag}{r['job']:<22}"
              f"{r['peptide_ca_rmsd']:>7.2f}{bb}{hv}{co}   {r['pred_vs_deposited']}")

    print(f"\n--- accounting ---")
    print(f"predictions with a crystal   : {len(J) - len(unmatched_pred)} of {len(J)}")
    for seq, name in unmatched_pred:
        print(f"    NO CRYSTAL for {name} ({seq})")
    print(f"crystals with a prediction   : {len(matched_xtal)} of {len(xtals)}")
    for x in unmatched_xtal:
        print(f"    NOT PREDICTED: {x['pdb']} {x['entity_seq']} ({x['source']})")
    assert len(matched_xtal) + len(unmatched_xtal) == len(xtals), "bucket mismatch"


if __name__ == "__main__":
    main()
