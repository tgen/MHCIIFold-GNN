#!/usr/bin/env python
"""Inventory the chains of each fetched crystal and assign each one a role.

    ~/micromamba/envs/bcell-repair/bin/python scripts/02_describe_crystals.py

Writes out/crystal_chains.tsv: one row per polymer chain with its auth chain id,
length, sequence, the deposit's own entity description, and a role
(peptide / mhc_a / mhc_b / tcr_a / tcr_b / other).

Why this exists: these deposits do NOT use the repo's canonical lettering
(A=peptide, B/C=MHC, D/E=TCR -- docs/TCRTRIFOLD.md). Getting the mapping wrong is a
QUIET failure: CDR RMSD still looks fine while every peptide and MHC-frame number is
garbage. So the mapping is written down, per structure, before any RMSD is computed.

Roles come from `_entity.pdbx_description` -- what the depositor said the chain IS --
not from sequence motifs. An earlier version guessed from motifs and confidently
labelled BOTH TCR chains of 1YMM and 8VCX as beta. A wrong label that looks decisive
is worse than no label, so anything the description cannot settle is UNASSIGNED and
the script exits non-zero.

Several entries hold more than one copy of the complex in the asymmetric unit
(6HBY x2, 8CMF x2, 8CME x3). Copies are numbered in `copy`; downstream code must say
which copy it used rather than silently taking the first.
"""
import sys, csv, re
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import protein_letters_3to1

HERE = Path(__file__).resolve().parent.parent
CIF = HERE / "out" / "crystals"
OUT = HERE / "out" / "crystal_chains.tsv"

# Order matters: the FIRST rule that fires wins, and length is checked before any
# description. In these deposits the only polymer of <=30 residues is the presented
# epitope, whereas descriptions are treacherous -- 2FSE calls its peptide chain
# "Collagen alpha-1(II)", which an earlier, looser TCR pattern matched on " alpha",
# labelling the peptide as a TCR chain and leaving the structure with no peptide at
# all. A confident wrong label is worse than no label.
DESC_ROLE = [
    ("mhc_a", re.compile(r"(DR|DQ|DP)[ -]?alpha|alpha[ -]?chain.*(class II|HLA-D)|"
                         r"class II.*(DR|E-K|I-E|I-A)[ -]?alpha|HLA-DRA|DQA1|DPA1", re.I)),
    ("mhc_b", re.compile(r"(DR|DQ|DP)[ -]?beta|beta[ -]?chain.*(class II|HLA-D)|"
                         r"class II.*(DRB|beta)|HLA-DRB|DQB1|DPB1|DRB1-\d|DR beta \d", re.I)),
    # 8VCX writes "T-CELL-RECEPTOR" (hyphenated); 4Y19/4Y1A name chains after the clone
    # only ("FS18_alpha"), so a bare suffix is allowed -- but ONLY as a whole-word
    # suffix on a token, never anywhere the word "alpha" happens to appear.
    ("tcr_a", re.compile(r"(T[ -]?cell[ -]?receptor|TCR).{0,40}alpha|"
                         r"alpha.{0,20}(TCR|T[ -]?cell)|TRAC|^\w+[_-]alpha$", re.I)),
    ("tcr_b", re.compile(r"(T[ -]?cell[ -]?receptor|TCR).{0,40}beta|"
                         r"beta.{0,20}(TCR|T[ -]?cell)|TRBC|^\w+[_-]beta$", re.I)),
    ("b2m",   re.compile(r"beta-?2[- ]microglobulin", re.I)),
]

# When a deposit names both MHC chains identically (1JK8 calls A and B both
# "MHC class II HLA-DQ8"), the description cannot separate them -- fall back to what
# the chain IS. These are the conserved N-terminal stretches of each chain class.
ANCHOR = {
    "mhc_a": ["EFDGDE", "FDGDEIFHVD", "HVIIQAEFYLNP", "EDIVADHVASYG", "IKEEHVIIQ",
              "GDTRPRFLE"[:0] or "EDIVADHV"],
    "mhc_b": ["GDTRPRFL", "NGTERVR", "RDSPEDFVYQFKG", "FLERYFHNQEE", "TERVRLLER",
              "DFVYQFKG"],
}

# Long chains that are legitimately none of the above. Listing them explicitly keeps
# them out of the UNASSIGNED bucket without loosening any role pattern.
OTHER = re.compile(r"exotoxin|superantigen|enterotoxin|lymphocyte activation gene|"
                   r"LAG-?3|antibody|nanobody|fab |lysozyme", re.I)


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


def seq_of(chain):
    """Residues that carry a CA -- waters and ligands have none, CIR does."""
    return "".join(res_letter(r.get_resname()) for r in chain if "CA" in r)

def chain_to_desc(path):
    """auth chain id -> entity description, straight from the mmCIF."""
    d = MMCIF2Dict(str(path))
    ids   = d.get("_entity.id", [])
    descs = d.get("_entity.pdbx_description", [])
    by_entity = dict(zip(ids, descs))
    out = {}
    e_ids  = d.get("_entity_poly.entity_id", [])
    strands = d.get("_entity_poly.pdbx_strand_id", [])
    for eid, strand in zip(e_ids, strands):
        for ch in str(strand).split(","):
            out[ch.strip()] = by_entity.get(eid, "")
    return out

def role_of(desc, seq):
    # 1. length first: the only short polymer in these deposits is the epitope
    if len(seq) <= 30:
        return "peptide"
    # 2. what the depositor says the chain is
    if desc:
        for role, rx in DESC_ROLE:
            if rx.search(desc):
                return role
    # 3. both MHC chains named the same -> decide from the sequence itself
    for role, anchors in ANCHOR.items():
        if any(a and a in seq for a in anchors):
            return role
    # 4. known non-players, named explicitly rather than by a loosened pattern
    if desc and OTHER.search(desc):
        return "other"
    return None

parser = MMCIFParser(QUIET=True)
rows, unassigned = [], []
cifs = sorted(CIF.glob("*.cif"))
if not cifs:
    sys.exit("no structures in out/crystals -- run 01_fetch_crystals.sh first")

for path in cifs:
    desc_by_chain = chain_to_desc(path)
    st = parser.get_structure(path.stem, str(path))
    seen = {}
    for chain in st[0]:
        seq = seq_of(chain)
        if not seq:
            continue
        desc = desc_by_chain.get(chain.id, "")
        role = role_of(desc, seq)
        if role is None:
            unassigned.append((path.stem, chain.id, len(seq), desc, seq[:40]))
            role = "UNASSIGNED"
        seen[role] = seen.get(role, 0) + 1
        rows.append(dict(pdb=path.stem, chain=chain.id, length=len(seq), role=role,
                         copy=seen[role], description=desc, sequence=seq))

# --- group copies by geometry, not by order of appearance -------------------------
# Copy numbers initially come from order of appearance, which assumes the nth peptide
# belongs to the nth MHC. That assumption is FALSE for 7NZE, where peptide EEE sits on
# the second MHC (24 A) and FFF on the first. So: define a copy by its MHC alpha chain,
# then assign every other chain to the copy whose alpha chain it is actually nearest.
import numpy as np

def centroid(model, chain_id):
    pts = [r["CA"].coord for r in model[chain_id] if r.id[0] == " " and "CA" in r]
    return np.array(pts).mean(axis=0)

for pdb in sorted({r["pdb"] for r in rows}):
    rs = [r for r in rows if r["pdb"] == pdb]
    alphas = [r for r in rs if r["role"] == "mhc_a"]
    if len(alphas) < 2:
        continue
    model = parser.get_structure(pdb, str(CIF / f"{pdb}.cif"))[0]
    anchors = {i: centroid(model, r["chain"]) for i, r in enumerate(alphas, 1)}
    for r in rs:
        if r["role"] == "mhc_a":
            r["copy"] = [i for i, a in enumerate(alphas, 1) if a["chain"] == r["chain"]][0]
            continue
        c = centroid(model, r["chain"])
        r["copy"] = min(anchors, key=lambda i: float(np.linalg.norm(c - anchors[i])))

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, lineterminator="\n", delimiter="\t",
        fieldnames=["pdb", "chain", "length", "role", "copy", "description", "sequence"])
    w.writeheader(); w.writerows(rows)

no_peptide = []
print(f"{len({r['pdb'] for r in rows})} structures, {len(rows)} polymer chains -> {OUT}")
for pdb in sorted({r["pdb"] for r in rows}):
    rs = [r for r in rows if r["pdb"] == pdb]
    peps = [r["copy"] for r in rs if r["role"] == "peptide"]
    if not peps:
        print(f"  {pdb}: NO PEPTIDE CHAIN -- " +
              " ".join(f"{r['chain']}={r['role']}" for r in rs), file=sys.stderr)
        no_peptide.append(pdb)
        continue
    ncopy = max(peps)
    tag = f"  [{ncopy} copies in the ASU]" if ncopy > 1 else ""
    print(f"  {pdb}: " + " ".join(f"{r['chain']}={r['role']}#{r['copy']}" for r in rs) + tag)

# --- verify the copy grouping is physical, not just positional --------------------
# Independent confirmation that the spatial regrouping above is self-consistent: each
# copy's peptide must be nearest its own MHC alpha chain. Intra-copy distances run
# 22-25 A and inter-copy 42-90 A, so the separation is unambiguous.
def centroid(model, chain_id):
    import numpy as np
    pts = [r["CA"].coord for r in model[chain_id] if r.id[0] == " " and "CA" in r]
    return np.array(pts).mean(axis=0)

import numpy as np
mismatched = []
for pdb in sorted({r["pdb"] for r in rows}):
    rs = [r for r in rows if r["pdb"] == pdb]
    ncopy = max([r["copy"] for r in rs if r["role"] == "peptide"] or [0])
    if ncopy < 2:
        continue
    model = parser.get_structure(pdb, str(CIF / f"{pdb}.cif"))[0]
    pep = {r["copy"]: r["chain"] for r in rs if r["role"] == "peptide"}
    mhc = {r["copy"]: r["chain"] for r in rs if r["role"] == "mhc_a"}
    for c, pch in pep.items():
        pc = centroid(model, pch)
        d = {k: float(np.linalg.norm(pc - centroid(model, v))) for k, v in mhc.items()}
        nearest = min(d, key=d.get)
        if nearest != c:
            mismatched.append((pdb, c, pch, nearest, d))
        print(f"    {pdb} peptide {pch} (copy{c}) -> nearest mhc_a is copy{nearest} "
              + " ".join(f"c{k}={v:.0f}A" for k, v in sorted(d.items())))
if mismatched:
    print("\nCOPY GROUPING IS WRONG for:", file=sys.stderr)
    for pdb, c, pch, near, d in mismatched:
        print(f"    {pdb} peptide {pch} labelled copy{c} but sits on copy{near}", file=sys.stderr)
    sys.exit("refusing to continue with a peptide assigned to the wrong MHC copy")

if no_peptide:
    print(f"\n{len(no_peptide)} structure(s) with no peptide chain: "
          + ", ".join(no_peptide), file=sys.stderr)
    sys.exit("refusing to continue: a class II entry with no peptide cannot be a reference")

if unassigned:
    print(f"\n{len(unassigned)} chain(s) could not be assigned a role:", file=sys.stderr)
    for pdb, ch, n, desc, head in unassigned:
        print(f"    {pdb} chain {ch} len={n} desc={desc!r} {head}...", file=sys.stderr)
    sys.exit("refusing to continue with an unassigned chain")
