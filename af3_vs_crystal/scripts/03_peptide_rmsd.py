#!/usr/bin/env python
"""Peptide CA RMSD after superposing on the MHC.

    ~/micromamba/envs/bcell-repair/bin/python scripts/03_peptide_rmsd.py \
        --ref out/crystals/8CMG.cif --pred /path/to/af3_model.cif

Reports four flavours of the same comparison -- CA, backbone (N CA C O), all heavy
atoms, and the CA of the 9-residue binding core -- all over the same residue pairs and
all in the MHC frame. See `paired_coords` and `binding_core` for what each covers.

The question is how well the peptide CONFORMATION AND POSE IN THE GROOVE are
predicted, so the two structures are brought into a common frame using the MHC only,
and the peptide RMSD is then computed WITHOUT any further superposition. Superposing
on the peptide itself would measure shape alone and hide a peptide that is correctly
shaped but sitting wrong in the groove.

This mirrors `analysis/triad_analysis/rmsd.py::true_pred_peptide_rmsd`, which is what
produced the manuscript's numbers:

  * CA atoms only (its comment notes backbone selection breaks on 5tez),
  * residues paired by sequence alignment with an infinite mismatch penalty, so only
    identical residues pair and a substitution drops out rather than pairing wrongly,
  * RMSD computed on the paired CAs with NO further superposition, i.e. the
    equivalent of `rms.rmsd(..., center=False, superposition=False)`.

Implemented with Biopython rather than MDAnalysis because the local env's MDAnalysis
has no mmCIF reader, and converting to PDB would lose chain ids longer than one
character (7NZF uses AAA/BBB/CCC).

It differs in one way, documented rather than hidden: rmsd.py moves each structure into
a canonical MHC frame precomputed upstream (`align_mhc_to_origin`), while this script
superposes the prediction directly onto the reference using the MHC CA pairs. For a
pairwise comparison the two are equivalent; for numbers that must sit in the same table
as the manuscript's, use rmsd.py on the cluster.

Chains are taken from out/crystal_chains.tsv for the reference and must be given
explicitly with --pred-chains for the prediction. There is no fallback guess: these
deposits use A=MHCa B=MHCb C=peptide while the repo canon is A=peptide B/C=MHC
(docs/TCRTRIFOLD.md), and a wrong mapping produces a plausible number, not an error.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import Bio.pairwise2
from Bio.PDB import MMCIFParser, PDBParser, Superimposer
from Bio.PDB.Polypeptide import protein_letters_3to1

HERE = Path(__file__).resolve().parent.parent
CHAINS = HERE / "out" / "crystal_chains.tsv"


def load_ref_chains(pdb_id, copy=1):
    """role -> chain id for one ASU copy, from the inventory 02 wrote."""
    if not CHAINS.exists():
        sys.exit(f"{CHAINS} missing -- run 02_describe_crystals.py first")
    out = {}
    with CHAINS.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["pdb"].upper() == pdb_id.upper() and int(r["copy"]) == copy:
                out[r["role"]] = r["chain"]
    if not out:
        sys.exit(f"no chains for {pdb_id} copy {copy} in {CHAINS}")
    for role in ("peptide", "mhc_a", "mhc_b"):
        if role not in out:
            sys.exit(f"{pdb_id} copy {copy} has no {role} chain -- cannot align")
    return out


def load(path):
    path = Path(path)
    parser = (MMCIFParser(QUIET=True) if path.suffix.lower() in (".cif", ".mmcif")
              else PDBParser(QUIET=True))
    return parser.get_structure(path.stem, str(path))[0]



# Modified residues are HETATM-flagged in mmCIF, so a plain `res.id[0] != " "` filter
# drops them -- and CIR (citrulline) is exactly the residue these RA epitopes exist to
# show (5JLZ, 5NIG, 6BIN). Dropping it would shorten the peptide silently and remove
# the position of interest from the RMSD. Membership in the polymer is therefore
# decided by "has a CA atom", which excludes waters and MPD without excluding CIR, and
# a modified residue is mapped to the parent amino acid it stands for so it pairs with
# the unmodified residue a prediction will contain.
MODIFIED = {
    "CIR": "R",  # citrulline <- arginine
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C", "HYP": "P",
    "MLY": "K", "M3L": "K", "ALY": "K", "AYA": "A", "NLE": "L", "ABA": "A",
    "PCA": "Q", "SAH": "C", "CME": "C", "OCS": "C", "KCX": "K", "LLP": "K",
}


def res_letter(resname):
    """One-letter code for a polymer residue, standard or modified."""
    if resname in protein_letters_3to1:
        return protein_letters_3to1[resname]
    return MODIFIED.get(resname, "X")


def ca_residues(model, chain_id):
    """CA atoms of one chain, in order. Membership is decided by the presence of a CA,
    which keeps modified residues (citrulline) and drops waters and ligands."""
    if chain_id not in model:
        avail = ",".join(c.id for c in model)
        sys.exit(f"chain {chain_id!r} not in structure (has: {avail})")
    out = [(res_letter(r.get_resname()), r["CA"]) for r in model[chain_id] if "CA" in r]
    if not out:
        sys.exit(f"chain {chain_id} has no CA atoms")
    return out


def pair_by_sequence(a, b):
    """Pair residues by global alignment; an infinite mismatch penalty means only
    identical residues pair, so a substituted position is dropped, never mispaired."""
    aln = Bio.pairwise2.align.globalms(
        "".join(x[0] for x in a), "".join(x[0] for x in b),
        2, -float("inf"), -2, -0.1,
    )[0]
    pa, pb = [], []
    i = j = 0
    for x, y in zip(aln.seqA, aln.seqB):
        if x != "-" and y != "-":
            if x == y:
                pa.append(a[i][1]); pb.append(b[j][1])
            i += 1; j += 1
        elif x != "-":
            i += 1
        else:
            j += 1
    return pa, pb


BACKBONE = ("N", "CA", "C", "O")
CORE_LEN = 9          # class II binding register
CONTACT_R = 5.0       # A; an atom this close to the MHC counts as a groove contact


# Side chains whose atom pairs are chemically equivalent: which of the two is called
# OD1 and which OD2 is arbitrary, so a name-by-name comparison can charge a model for a
# labelling difference rather than a structural one. When SYMMETRIC is on, each such
# residue is compared under both labellings and the lower deviation is kept.
SYMMETRIC_PAIRS = {
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "ARG": [("NH1", "NH2")],
}
SYMMETRIC = False     # measured before being switched on; see the audit note in README


def paired_coords(ra, pb, atoms=None):
    """Coordinates of the atoms shared by each paired residue.

    Atom names are INTERSECTED per residue rather than assumed identical, so a modified
    residue works: citrulline pairs with the arginine a prediction contains, and the
    atoms they genuinely have in common (N, CA, C, O, CB, CG, CD, NE ...) are compared
    while the ones only one of them has are dropped. A substituted position never
    reaches here at all -- pair_by_sequence keeps identical residues only.
    """
    R, P = [], []
    for r_res, p_res in zip(ra, pb):
        names = {a.get_id() for a in r_res if a.element != "H"} & \
                {a.get_id() for a in p_res if a.element != "H"}
        if atoms is not None:
            names &= set(atoms)
        names = sorted(names)
        rc = [r_res[nm].coord for nm in names]
        pc = [p_res[nm].coord for nm in names]
        swaps = SYMMETRIC_PAIRS.get(p_res.get_resname(), []) if SYMMETRIC else []
        if swaps:
            alt = list(names)
            for x, y in swaps:
                if x in alt and y in alt:
                    i, j = alt.index(x), alt.index(y)
                    alt[i], alt[j] = alt[j], alt[i]
            pc_alt = [p_res[nm].coord for nm in alt]
            d0 = sum(float(np.sum((np.array(a) - np.array(b)) ** 2)) for a, b in zip(rc, pc))
            d1 = sum(float(np.sum((np.array(a) - np.array(b)) ** 2)) for a, b in zip(rc, pc_alt))
            if d1 < d0:
                pc = pc_alt
        R += rc; P += pc
    return np.array(R), np.array(P)


def rmsd_of(R, P):
    if len(R) == 0:
        return None, 0
    d = np.linalg.norm(P - R, axis=1)
    return float(np.sqrt((d ** 2).mean())), len(R)


def binding_core(ref_model, pep_residues, mhc_chain_ids):
    """Indices of the CORE_LEN-residue window most buried in the groove.

    Class II peptides extend out of both ends of the groove and their termini are the
    least restrained part of the structure -- several of the worst deviations in this
    set are terminal. The core is found from the geometry rather than assumed to be
    centred: for each residue, count MHC atoms within CONTACT_R, then take the
    contiguous window of CORE_LEN with the most contacts.
    """
    mhc = [a for cid in mhc_chain_ids for r in ref_model[cid] for a in r
           if a.element != "H"]
    if not mhc or len(pep_residues) < CORE_LEN:
        return None
    mx = np.array([a.coord for a in mhc])
    counts = []
    for res in pep_residues:
        pa = np.array([a.coord for a in res if a.element != "H"])
        d = np.linalg.norm(mx[None, :, :] - pa[:, None, :], axis=2)
        counts.append(int((d < CONTACT_R).sum()))
    counts = np.array(counts)
    best = max(range(len(counts) - CORE_LEN + 1),
               key=lambda i: counts[i:i + CORE_LEN].sum())
    return best, best + CORE_LEN


GROOVE_R = 12.0       # A around the peptide that counts as the binding platform


def peptide_rmsd(ref_path, pred_path, ref_chains, pred_chains, label="", frame="full"):
    """frame: "full"   superpose on every paired MHC CA (both domains)
              "groove" superpose only on MHC residues within GROOVE_R of the peptide

    Why the choice matters. Class II MHC is a peptide-binding platform (alpha1+beta1)
    sitting on two Ig-like domains, and those domains hinge. Fitting on the WHOLE
    molecule lets that hinge set the frame the peptide is then measured in: across this
    set the full-MHC fit ranges 0.39-5.56 A, and 8CMF's two ASU copies fit at 0.39 and
    3.97 A for the SAME prediction. Any such rotation moves the groove and is charged to
    the peptide, which did not move. Superposing on the platform alone removes that.
    """
    ref = load(ref_path)
    pred = load(pred_path)

    # --- superpose on the MHC ----------------------------------------------
    ref_mhc, pred_mhc = [], []
    for role in ("mhc_a", "mhc_b"):
        ra, pb = pair_by_sequence(ca_residues(ref, ref_chains[role]),
                                  ca_residues(pred, pred_chains[role]))
        ref_mhc += ra; pred_mhc += pb
    if len(ref_mhc) != len(pred_mhc) or len(ref_mhc) < 50:
        sys.exit(f"MHC pairing failed: {len(ref_mhc)} vs {len(pred_mhc)} CA")

    if frame == "groove":
        pep_xyz = np.array([a.coord for _, a in ca_residues(ref, ref_chains["peptide"])])
        keep = [i for i, a in enumerate(ref_mhc)
                if np.linalg.norm(pep_xyz - a.coord, axis=1).min() <= GROOVE_R]
        if len(keep) < 30:
            sys.exit(f"only {len(keep)} MHC CA within {GROOVE_R} A of the peptide")
        ref_mhc = [ref_mhc[i] for i in keep]
        pred_mhc = [pred_mhc[i] for i in keep]
    elif frame != "full":
        sys.exit(f"unknown frame {frame!r} (use full or groove)")

    sup = Superimposer()
    sup.set_atoms(ref_mhc, pred_mhc)          # (fixed, moving)
    sup.apply(list(pred.get_atoms()))         # move the WHOLE prediction

    # --- peptide, in that frame, with NO further superposition --------------
    rp = ca_residues(ref, ref_chains["peptide"])
    pp = ca_residues(pred, pred_chains["peptide"])
    rca, pca = pair_by_sequence(rp, pp)
    if len(rca) != len(pca) or not rca:
        sys.exit(f"peptide pairing failed: {len(rca)} vs {len(pca)} CA")

    r = np.array([a.coord for a in rca])
    p = np.array([a.coord for a in pca])
    dev = np.linalg.norm(p - r, axis=1)
    val = float(np.sqrt((dev ** 2).mean()))

    # --- the same comparison over wider atom sets -------------------------
    # Residue pairs behind the CA numbers, so every metric covers the same residues.
    ref_res = [a.get_parent() for a in rca]
    pred_res = [a.get_parent() for a in pca]
    bb, n_bb = rmsd_of(*paired_coords(ref_res, pred_res, BACKBONE))
    heavy, n_heavy = rmsd_of(*paired_coords(ref_res, pred_res))

    core = binding_core(ref, ref_res, [ref_chains["mhc_a"], ref_chains["mhc_b"]])
    if core:
        i, j = core
        core_ca, n_core = rmsd_of(np.array([a.coord for a in rca[i:j]]),
                                  np.array([a.coord for a in pca[i:j]]))
        core_span = f"{i + 1}-{j}"
    else:
        core_ca, n_core, core_span = None, 0, ""

    return dict(label=label, peptide_rmsd=val, n_pep_ca=len(rca),
                backbone_rmsd=bb, n_backbone=n_bb,
                heavy_rmsd=heavy, n_heavy=n_heavy,
                core_ca_rmsd=core_ca, n_core=n_core, core_span=core_span,
                mhc_align_rmsd=float(sup.rms), n_mhc_ca=len(ref_mhc),
                ref_pep_len=len(rp), pred_pep_len=len(pp),
                max_dev=float(dev.max()), per_residue=dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="crystal structure (mmCIF/PDB)")
    ap.add_argument("--pred", required=True, help="predicted structure (mmCIF/PDB)")
    ap.add_argument("--ref-copy", type=int, default=1, help="ASU copy of the reference")
    ap.add_argument("--ref-chains", help="override, as peptide,mhc_a,mhc_b")
    ap.add_argument("--pred-chains", required=True,
                    help="prediction chains as peptide,mhc_a,mhc_b (no default: the "
                         "repo canon A=peptide differs from these deposits' C=peptide)")
    ap.add_argument("--frame", choices=("full", "groove"), default="full",
                    help="superpose on the whole MHC (default) or only the binding "
                         "platform within 12 A of the peptide")
    ap.add_argument("--per-residue", action="store_true")
    a = ap.parse_args()

    if a.ref_chains:
        p, ma, mb = a.ref_chains.split(",")
        ref_chains = {"peptide": p, "mhc_a": ma, "mhc_b": mb}
    else:
        ref_chains = load_ref_chains(Path(a.ref).stem, a.ref_copy)
    p, ma, mb = a.pred_chains.split(",")
    pred_chains = {"peptide": p, "mhc_a": ma, "mhc_b": mb}

    r = peptide_rmsd(a.ref, a.pred, ref_chains, pred_chains, frame=a.frame,
                     label=f"{Path(a.ref).stem}#{a.ref_copy} vs {Path(a.pred).stem}")
    print(f"{r['label']}")
    print(f"  ref chains  peptide={ref_chains['peptide']} "
          f"mhc_a={ref_chains['mhc_a']} mhc_b={ref_chains['mhc_b']}")
    print(f"  pred chains peptide={pred_chains['peptide']} "
          f"mhc_a={pred_chains['mhc_a']} mhc_b={pred_chains['mhc_b']}")
    print(f"  MHC superposition : {r['mhc_align_rmsd']:.3f} A over {r['n_mhc_ca']} CA "
          f"[{a.frame} frame]")
    print(f"  PEPTIDE CA RMSD   : {r['peptide_rmsd']:.3f} A over {r['n_pep_ca']} CA "
          f"(ref {r['ref_pep_len']}, pred {r['pred_pep_len']} modelled)")
    if r["backbone_rmsd"] is not None:
        print(f"  backbone (N CA C O): {r['backbone_rmsd']:.3f} A over {r['n_backbone']} atoms")
    if r["heavy_rmsd"] is not None:
        print(f"  all heavy atoms   : {r['heavy_rmsd']:.3f} A over {r['n_heavy']} atoms")
    if r["core_ca_rmsd"] is not None:
        print(f"  9-mer core CA     : {r['core_ca_rmsd']:.3f} A (peptide residues "
              f"{r['core_span']})")
    print(f"  worst residue     : {r['max_dev']:.3f} A")
    if a.per_residue:
        print("  per-residue deviation (A): " +
              " ".join(f"{d:.2f}" for d in r["per_residue"]))


if __name__ == "__main__":
    main()
