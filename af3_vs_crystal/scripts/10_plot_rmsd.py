#!/usr/bin/env python
"""Ranked dot plot of peptide Ca RMSD, one row per crystal, labelled by PDB.

    ~/micromamba/envs/bcell-repair/bin/python scripts/10_plot_rmsd.py

In:  out/rmsd_summary.tsv, out/peptide_rmsd_all_samples.tsv, out/asu_noise_floor.tsv
Out: out/figures/rmsd_dotplot.png

Four panels, one per metric (CA / backbone / heavy atom / 9-mer core), each against
its OWN copy-to-copy floor.

Form: the job is "compare magnitude across a handful of named things", so this is a
ranked dot plot, not a bar chart (no zero-anchored area to compare) and not a scatter
of two measures (there is only one measure). Each row is one crystal/ASU-copy pair,
sorted best to worst, labelled with its PDB id.

What each row shows:
  * filled dot   = MEDIAN over all 50 AF3 models (10 seeds x 5 samples)
  * open marker  = the TOP-RANKED model, i.e. what you get if you take AF3's own pick
  * thin line    = 5th-95th percentile of the 50 models

Every model is drawn. The faint outer line is the FULL range of all 50 models, the
heavier inner bar the 5th-95th percentile, the dot the median and the tick the mean.
An earlier version drew only the percentile bar, which cropped the models that place the
peptide right out of the groove off the axis -- they were reported as a count elsewhere,
but a reader of the figure could not see them. They are the most interesting models in
the set (see FAILURES.md) and they are now visible. Where the maximum runs past the axis
the line ends in an arrow carrying its value.

Styled to match the repo's other figures (figures/**/*.ipynb): plain matplotlib
defaults, #3a78b8 marks, large axis labels and ticks, top/right spines hidden, and NO
figure title -- the caption belongs in the manuscript, not burned into the PNG.

The grey band is the crystallographic noise floor from 04_asu_noise_floor.py: the same
measurement between independent copies of one crystal. A point inside that band is not
"accurate to 1 A", it is indistinguishable from the experiment's own spread.
"""
import csv, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent.parent
SUMMARY = HERE / "out" / "rmsd_summary.tsv"
ALL = HERE / "out" / "peptide_rmsd_all_samples.tsv"
FLOOR = HERE / "out" / "asu_noise_floor.tsv"
OUT = HERE / "out" / "figures" / "rmsd_dotplot.png"

# House style, matching the repo's other figures (figures/**/*.ipynb): plain
# matplotlib defaults, #3a78b8 marks, large axis labels, only top/right spines hidden.
MEDIAN_C, TOP_C = "#3a78b8", "#d1662a"
MUTED, FAINT, BAND = "0.45", "0.72", "0.88"
MEAN_C = "#d1662a"
XMAX = 4.0


def main():
    for p in (SUMMARY, ALL, FLOOR):
        if not p.exists():
            sys.exit(f"{p} missing -- run 05/06 (and 04) first")

    rows = list(csv.DictReader(SUMMARY.open(), delimiter="\t"))
    vals = defaultdict(list)
    for r in csv.DictReader(ALL.open(), delimiter="\t"):
        for col in ("peptide_ca_rmsd", "backbone_rmsd", "heavy_rmsd", "core_ca_rmsd"):
            if r.get(col):
                vals[(r["pdb"], r["copy"], col)].append(float(r[col]))
    floor_rows = list(csv.DictReader(FLOOR.open(), delimiter="\t"))

    # Order: crystals ranked by their best copy, ASU copies kept together. Rows whose
    # prediction and crystal are NOT the same peptide (dagger) go last, below a divider,
    # so they are never read as part of the main ranking.
    best_of = {}
    for r in rows:
        best_of[r["pdb"]] = min(best_of.get(r["pdb"], 9e9), float(r["median"]))
    dag = lambda r: r.get("caveat") == "dagger"
    # pdb is in the key so two crystals with the SAME best median cannot interleave;
    # the list is reversed below because row 0 is drawn at the BOTTOM of the axes
    rows.sort(key=lambda r: (dag(r), best_of[r["pdb"]], r["pdb"], int(r["copy"])))
    rows = rows[::-1]
    n = len(rows)

    # y positions: a small gap between different crystals, so ASU copies read as a group
    ypos, y, prev = [], 0.0, None
    for r in rows:
        if prev is not None and r["pdb"] != prev:
            y += 0.45
        ypos.append(y); y += 1.0; prev = r["pdb"]
    ytop = ypos[-1]
    n_dag = sum(1 for r in rows if dag(r))

    # One panel per metric, each against its OWN copy-to-copy floor: a core number
    # judged against a whole-peptide floor would look far better than it is. The x
    # scale is SHARED so the panels can be compared by eye.
    PANELS = [("peptide_ca_rmsd", "median",          "Whole peptide Cα"),
              ("backbone_rmsd",   "median_backbone", "Backbone (N, CA, C, O)"),
              ("heavy_rmsd",      "median_heavy",    "All heavy atoms"),
              ("core_ca_rmsd",    "median_core_ca",  "9-mer binding core Cα")]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.9 * len(PANELS) + 1.6,
                                                      0.40 * n + 2.6),
                             dpi=300, sharey=True, gridspec_kw={"wspace": 0.22})

    for k, (ax, (col, medcol, title)) in enumerate(zip(axes, PANELS)):
        # alternate shading per CRYSTAL (not per row) so copies share a band
        shade, prev = False, None
        for i, r in enumerate(rows):
            if r["pdb"] != prev:
                shade = not shade; prev = r["pdb"]
            if shade:
                ax.axhspan(ypos[i] - 0.5, ypos[i] + 0.5, color="0.965", lw=0, zorder=0)

        fl = [float(r[col]) for r in floor_rows if r.get(col) not in (None, "")]
        flo, fhi = min(fl), max(fl)
        ax.axvspan(flo, fhi, color=BAND, zorder=0.5, lw=0, alpha=0.75)

        for i, r in enumerate(rows):
            yy = ypos[i]
            v = np.array(sorted(vals[(r["pdb"], r["copy"], col)]))
            if not len(v):
                continue
            lo_, hi_ = float(v.min()), float(v.max())
            p5, p95 = np.percentile(v, 5), np.percentile(v, 95)
            # FULL range, lighter, so models that put the peptide right out of the
            # groove stay visible; the darker inner bar is the 5th-95th percentile
            ax.plot([lo_, min(hi_, XMAX)], [yy, yy], color=FAINT, lw=1.1, zorder=1,
                    solid_capstyle="butt", clip_on=False)
            ax.plot([p5, min(p95, XMAX)], [yy, yy], color=MUTED, lw=3.0, zorder=2,
                    solid_capstyle="butt", clip_on=False)
            if hi_ > XMAX:                  # off-scale: an arrow and the true value
                ax.plot([XMAX], [yy], marker=">", ms=5, color=FAINT, zorder=2,
                        clip_on=False)
                ax.annotate(f"{hi_:.0f}", xy=(XMAX + 0.08, yy), ha="left", va="center",
                            fontsize=9, color=MUTED, annotation_clip=False)
            mean_ = float(v.mean())
            if mean_ <= XMAX:               # mean AND median, so neither is a hidden choice
                ax.plot([mean_], [yy], marker="|", ms=12, mew=2.2, color=MEAN_C, zorder=3)
            med = float(r[medcol]) if r.get(medcol) not in (None, "") else None
            if med is not None:
                ax.plot([med], [yy], marker="o", ms=9.5, color=MEDIAN_C, mec="white",
                        mew=1.3, zorder=4)

        if n_dag:                           # divider above the daggered rows
            ydiv = (ypos[n_dag - 1] + ypos[n_dag]) / 2
            ax.axhline(ydiv, color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1)

        ax.set_xlim(0, XMAX)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_title(title, fontsize=16, pad=30)
        ax.text(-0.02, 1.075, "ABCD"[k], transform=ax.transAxes, fontsize=20,
                fontweight="bold", ha="left", va="bottom")
        ax.set_xlabel("RMSD (Å)", fontsize=15, labelpad=8)
        ax.tick_params(axis="x", labelsize=13, length=5, width=1.1)
        ax.tick_params(axis="y", length=0, pad=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.annotate(f"crystal copy-to-copy {flo:.2f}–{fhi:.2f} Å",
                    xy=((flo + fhi) / 2, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 5), textcoords="offset points",
                    ha="left" if fhi < 1.0 else "center", va="bottom",
                    fontsize=10, color="0.35")

    ncopies = {}
    for r in rows:
        ncopies[r["pdb"]] = ncopies.get(r["pdb"], 0) + 1
    labels = [("† " if dag(r) else "") + r["pdb"]
              + (f"  copy {r['copy']}" if ncopies[r["pdb"]] > 1 else "") for r in rows]
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(labels, fontsize=13)
    axes[0].set_ylim(-0.6, ytop + 0.6)

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=9.5, color=MEDIAN_C, mec="white", mew=1.3,
               label="median of 50 models"),
        Line2D([], [], marker="|", ls="", ms=12, mew=2.2, color=MEAN_C, label="mean"),
        Line2D([], [], color=MUTED, lw=3.0, label="5th–95th percentile"),
        Line2D([], [], color=FAINT, lw=1.1, label="full range (arrow + value if > 4 Å)"),
        Patch(facecolor=BAND, alpha=0.75, label="crystal copy-to-copy range"),
        Line2D([], [], ls="", label="†  prediction and crystal differ by substitutions"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
        fontsize=12.5, handletextpad=0.6, columnspacing=2.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"{n} rows -> {OUT}")
    for col, medcol, name in (("peptide_ca_rmsd", "median", "whole peptide CA"),
                              ("backbone_rmsd", "median_backbone", "backbone"),
                              ("heavy_rmsd", "median_heavy", "heavy atoms"),
                              ("core_ca_rmsd", "median_core_ca", "9-mer core CA")):
        fl = [float(r[col]) for r in floor_rows if r.get(col) not in (None, "")]
        hi = max(fl)
        inside = [r for r in rows if r.get(medcol) not in (None, "")
                  and float(r[medcol]) <= hi]
        print(f"{name:<17}: {len(inside)} of {n} inside the copy-to-copy range "
              f"(<= {hi:.2f} A)")


if __name__ == "__main__":
    main()
