#!/usr/bin/env python3
"""Blinded percentile on the frozen equalized pool (pool_v1, 1,933 drugs):
TxGNN (graph) vs TxGemma-predict (transformer, post-LoRA) vs a Tanimoto 5-NN baseline,
matched reference drug per disease, with TxGNN's specificity control shown.

Three things this figure exists to prevent, all of which the first version allowed:
  1. reading a low TxGemma bar as "the model works" when the untrained base model was
     better (mastocytosis: 14.6 zero-shot -> 77.9 fine-tuned on the decoy pool);
  2. reading a low TxGNN bar as a disease-level hit when another disease's model ranks
     the same drug better (tafamidis: 155 under TTR's own model, 8 under Fabry's);
  3. reading either bar as good without the chemical baseline that beats them both
     (mastocytosis: midostaurin is 11th of 1,933 on Morgan-2 fingerprints alone).
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LIB = "/projects/bisa/sshah21/library"
T25 = f"{LIB}/03_txgnn/try25_23-11-2026_equalized-pool"
GNN = f"{T25}/results_pool_v1_ensemble.csv"   # seed-ensembled + specificity columns
GNNS = f"{T25}/results_pool_v1.csv"           # per seed
GEM = ("/work/hdd/bisa/sshah21/results/txgemma_lora_orphan/"
       "try16_equalized_pool/results_pool_v1.csv")
OUT = f"{LIB}/_figures"
N_POOL = 1933
CHANCE = 49.902

# disease -> (txgnn eval_node, txgnn ref_drug, txgemma target, txgemma path, blank reason)
PAIRS = [
    ("Malignant hyperthermia", "exercise-induced malignant hyperthermia",
                                                   "Dantrolene",  None,           None,       "no TxGemma arm"),
    ("Gaucher",                "Gaucher disease",  "miglustat",   "gaucher",      None,       "reference not in pool"),
    ("Fabry",                  "Fabry disease",    "migalastat",  "fabry",        "crossval", None),
    ("ATTR amyloidosis",       "ATTRV122I amyloidosis",
                                                   "tafamidis",   "attr",         "stepstop", None),
    ("Phenylketonuria",        "phenylketonuria",  "Sapropterin", None,           None,       "no TxGemma arm"),
    ("Cystic fibrosis",        "cystic fibrosis",  "lumacaftor",  "cf",           "stepstop", None),
    ("Tyrosinemia",            "tyrosinemia",      "Nitisinone",  "tyrosinemia",  "stepstop", None),
    ("Chronic eos. leukemia",  "chronic eosinophilic leukemia",
                                                   "Imatinib",    "cel",          "stepstop", None),
    ("Systemic mastocytosis",  "systemic mastocytosis",
                                                   "Midostaurin", "mastocytosis", "stepstop", None),
    ("NSIAD",                  "nephrogenic syndrome of inappropriate antidiuresis",
                                                   "Tolvaptan",   "nsiad",        None,       "reference not in pool"),
]

# ---- TxGNN: ensembled rank + the specificity control that sits beside it
gnn = {}
for r in csv.DictReader(open(GNN)):
    gnn[(r["eval_node"], r["ref_drug"])] = r

# ---- TxGNN: per-seed ranks.  The bar is NOT the mean of these: try25 ensembles by
# ranking the MEAN RAW SCORE, so the whisker is a spread around a differently-computed
# centre.  try11 established the variance is inherent embedding stochasticity.
gnn_seeds = {}
for r in csv.DictReader(open(GNNS)):
    if r["rank_pool"]:
        gnn_seeds.setdefault((r["eval_node"], r["ref_drug"]), []).append(
            float(r["rank_pool"]) / N_POOL * 100.0)

# ---- TxGemma: post-LoRA, blinded, digit_exp, per seed; and the Tanimoto 5-NN baseline
# carried in the same rows.  The kNN is a property of the disease + pool + actives list,
# not of the arm, so it is the baseline for BOTH bars on that row.
gem, knn, triv = {}, {}, {}
for r in csv.DictReader(open(GEM)):
    if r["readout"] == "digit_exp" and r["percentile_pool"]:
        key = (r["target"], r["path_used"])
        gem.setdefault(key, []).append(float(r["percentile_pool"]))
        if r["tanimoto_percentile_pool"]:
            knn[key] = float(r["tanimoto_percentile_pool"])
            triv[key] = (r["trivial_flag"].strip().lower() == "true")

rows = []
for label, node, drug, tgt, path, blank in PAIRS:
    g = gnn.get((node, drug), {})
    gp = (float(g["rank_pool"]) / N_POOL * 100.0) if g.get("rank_pool") else None
    gs = gnn_seeds.get((node, drug), [])
    seeds = gem.get((tgt, path), []) if path else []
    rows.append(dict(
        label=label, drug=drug, blank=blank, path=path, tgt=tgt,
        gnn=gp, gnn_lo=(min(gs) if gs else None), gnn_hi=(max(gs) if gs else None),
        spec=(g.get("specificity_pass", "") == "True"),
        other_rank=(float(g["best_other_rank"]) if g.get("best_other_rank") else None),
        other_model=(g.get("best_other_model") or "").split("/")[0],
        gem=(float(np.mean(seeds)) if seeds else None),
        gem_lo=(min(seeds) if seeds else None), gem_hi=(max(seeds) if seeds else None),
        n_seeds=len(seeds),
        knn=knn.get((tgt, path)) if path else None,
        knn_trivial=triv.get((tgt, path), False) if path else False))
rows.sort(key=lambda d: (d["gnn"] is None, d["gnn"] if d["gnn"] is not None else 0))

# ---- plot
C_GNN, C_GEM, C_NA, C_KNN = "#2b6ca3", "#4c8c40", "#bfbfbf", "#b02020"
y = np.arange(len(rows))[::-1]
h = 0.34
fig, ax = plt.subplots(figsize=(12.4, 7.8))
fig.subplots_adjust(left=0.135, right=0.80, top=0.93, bottom=0.345)

ax.axvline(CHANCE, color="0.25", ls="--", lw=1.3, zorder=5)
ax.text(CHANCE + 0.8, y.max() + 0.62, f"chance {CHANCE:.1f}%", color="0.25", fontsize=8.5)

for yi, d in zip(y, rows):
    # ---- TxGNN, hatched when it fails the specificity control
    if d["gnn"] is not None:
        fails = not d["spec"]
        ax.barh(yi + h/2, d["gnn"], height=h, color=C_GNN, zorder=3,
                hatch=("////" if fails else None),
                edgecolor=("white" if fails else "none"), linewidth=0)
        if d["gnn_lo"] is not None:
            ax.plot([d["gnn_lo"], d["gnn_hi"]], [yi + h/2]*2, color="0.15", lw=1.3, zorder=4)
            for v in (d["gnn_lo"], d["gnn_hi"]):
                ax.plot([v, v], [yi + h/2 - .08, yi + h/2 + .08], color="0.15", lw=1.3, zorder=4)
        ax.text(max(d["gnn_hi"] or 0, d["gnn"]) + 1.8, yi + h/2, f'{d["gnn"]:.1f}',
                va="center", fontsize=8.5, color=C_GNN, fontweight="bold", zorder=8,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    else:
        ax.barh(yi + h/2, 100, height=h, color=C_NA, alpha=.22, zorder=2)
        ax.text(1.2, yi + h/2, "reference not in pool", va="center", fontsize=8,
                color="0.35", style="italic")
    # ---- TxGemma (post-LoRA)
    if d["gem"] is not None:
        ax.barh(yi - h/2, d["gem"], height=h, color=C_GEM, zorder=3)
        ax.plot([d["gem_lo"], d["gem_hi"]], [yi - h/2]*2, color="0.15", lw=1.3, zorder=4)
        for v in (d["gem_lo"], d["gem_hi"]):
            ax.plot([v, v], [yi - h/2 - .08, yi - h/2 + .08], color="0.15", lw=1.3, zorder=4)
        ax.text(max(d["gem_hi"], d["gem"]) + 1.8, yi - h/2, f'{d["gem"]:.1f}',
                va="center", fontsize=8.5, color=C_GEM, fontweight="bold", zorder=8,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    else:
        ax.barh(yi - h/2, 100, height=h, color=C_NA, alpha=.22, zorder=2)
        ax.text(1.2, yi - h/2, d["blank"] or "not evaluable", va="center", fontsize=8,
                color="0.35", style="italic")
    # ---- Tanimoto 5-NN: one per disease row, the baseline for both bars
    if d["knn"] is not None:
        ls = ":" if d["knn_trivial"] else "-"
        ax.plot([d["knn"]]*2, [yi - 0.40, yi + 0.40], color=C_KNN, lw=2.0,
                ls=ls, zorder=6, solid_capstyle="butt")
        ax.plot([d["knn"]], [yi], marker="D", ms=5.5, color=C_KNN, zorder=7)
        note = f'{d["knn"]:.1f}' + (" trivial" if d["knn_trivial"] else "")
        # label sits on the row's own centre line, so it cannot be read as the row above
        ax.text(d["knn"] + 1.4, yi, note, color=C_KNN, fontsize=7.8,
                ha="left", va="center", fontweight="bold", zorder=8,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))

# ---- specificity column, outside the axes
ax.text(103, y.max() + 0.95, "TxGNN specificity control", fontsize=8.5,
        fontweight="bold", color="0.25", clip_on=False)
for yi, d in zip(y, rows):
    if d["gnn"] is None:
        txt, col = "—", "0.6"
    elif d["spec"]:
        txt, col = "pass", "#1a7a1a"
    else:
        txt = f'FAIL · {int(d["other_rank"])} under {d["other_model"]}'
        col = "#b02020"
    ax.text(103, yi, txt, fontsize=8, color=col, va="center", clip_on=False)

ax.set_yticks(y)
ax.set_yticklabels([f'{d["label"]}\n{d["drug"].lower()}' for d in rows], fontsize=9)
ax.set_xlim(0, 100)
ax.set_ylim(y.min() - 0.8, y.max() + 0.8)
ax.set_xlabel("Blinded percentile of the reference drug among 1,933 approved drugs"
              "  (lower = better)", fontsize=9.5)
ax.set_title("Equalized pool (pool_v1, sha256 9d89c3ce…): graph vs transformer vs "
             "chemical baseline, blinded", fontsize=11.5, pad=14)
ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.grid(axis="x", color="0.9", lw=.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)

ax.legend(handles=[
    Patch(color=C_GNN, label="TxGNN (graph; bar = seed-ensembled rank, whisker = per-seed range)"),
    Patch(facecolor=C_GNN, hatch="////", edgecolor="white",
          label="        hatched = FAILS the specificity control (another disease's model ranks it better)"),
    Patch(color=C_GEM, label="TxGemma-predict (transformer, AFTER LoRA; digit_exp, bar = mean of 3 seeds)"),
    Line2D([0], [0], color=C_KNN, lw=1.9, marker="v", ms=6,
           label="Tanimoto 5-NN baseline (Morgan-2, no model; dotted = leak-flagged, inadmissible)"),
    Patch(color=C_NA, alpha=.22, label="no transformer arm / reference not in pool")],
    loc="upper left", bbox_to_anchor=(0.0, -0.125), ncol=1,
    fontsize=8.3, frameon=False, handlelength=2.2, borderpad=0)

fig.text(0.012, 0.012,
         "TxGNN: try25 (pool rank / 1,933, seed-ensembled).  TxGemma-predict: try16, blinded digit_exp, post-LoRA "
         "(fabry = 6-fold crossval path; others = step-stopping).\n"
         "The two bars are not the same statistic: TxGNN's is the rank of the mean raw score over 3 seeds, "
         "TxGemma's is the mean of 3 per-seed percentiles.  Both whiskers are per-seed min-max.\n"
         "The kNN is a property of the disease, pool and actives list rather than of either arm, so it is the "
         "baseline for both bars on its row.  It needs measured bioactivity against the\n"
         "target, which exists for 6 of the 10 diseases: MH and PKU have no actives set, and gaucher/nsiad are "
         "not rankable on this pool.  No pre-LoRA bar can be drawn — zero-shot was\n"
         "only ever measured on the old 8-target decoy pool, where LoRA moved mastocytosis 14.6 → 77.9 and "
         "cel 94.2 → 20.6, i.e. this figure's green bars are worse than the untrained\n"
         "base model on some rows and far better on others.  MH = exercise-induced node (its specificity pass is "
         "against the sibling anaesthesia node, 11 vs 29).  ATTR = ATTRV122I node.",
         fontsize=7.0, color="0.4", va="bottom", linespacing=1.5)

os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/equalized_pool_blinded_gnn_vs_gemma.{ext}", dpi=220)

# ---- matching data table
with open(f"{OUT}/equalized_pool_blinded_gnn_vs_gemma.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["disease", "reference_drug",
                "txgnn_percentile_ensembled", "txgnn_seed_min", "txgnn_seed_max",
                "txgnn_specificity_pass", "txgnn_best_rank_under_other_model",
                "txgnn_other_model",
                "txgemma_lora_percentile_mean", "txgemma_seed_min", "txgemma_seed_max",
                "txgemma_n_seeds", "txgemma_path",
                "tanimoto_knn_percentile", "tanimoto_knn_trivial_flag"])
    for d in rows:
        w.writerow([
            d["label"], d["drug"],
            f'{d["gnn"]:.3f}' if d["gnn"] is not None else "NOT_IN_POOL",
            f'{d["gnn_lo"]:.3f}' if d["gnn_lo"] is not None else "",
            f'{d["gnn_hi"]:.3f}' if d["gnn_hi"] is not None else "",
            ("pass" if d["spec"] else "FAIL") if d["gnn"] is not None else "",
            int(d["other_rank"]) if d["other_rank"] else "",
            d["other_model"] or "",
            f'{d["gem"]:.3f}' if d["gem"] is not None else
            (d["blank"] or "PENDING").upper().replace(" ", "_"),
            f'{d["gem_lo"]:.3f}' if d["gem"] is not None else "",
            f'{d["gem_hi"]:.3f}' if d["gem"] is not None else "",
            d["n_seeds"] or "", d["path"] or "",
            # two distinct reasons for a missing kNN: no bioactivity dataset for the
            # target at all (mh, pku), vs. a dataset that exists but cannot rank this
            # reference because it is absent from pool_v1 (gaucher, nsiad)
            f'{d["knn"]:.3f}' if d["knn"] is not None else
            ("NO_ACTIVES_SET" if d["tgt"] is None else "REFERENCE_NOT_IN_POOL"),
            ("TRIVIAL" if d["knn_trivial"] else "") if d["knn"] is not None else ""])

print(f'{"disease":24s} {"TxGNN":>7s} {"spec":>5s} {"LoRA":>7s} {"kNN":>8s}  beats both?')
for d in rows:
    g = "—" if d["gnn"] is None else f'{d["gnn"]:.1f}'
    m = "—" if d["gem"] is None else f'{d["gem"]:.1f}'
    k = "—" if d["knn"] is None else f'{d["knn"]:.1f}'
    sp = "—" if d["gnn"] is None else ("pass" if d["spec"] else "FAIL")
    win = ""
    if d["knn"] is not None and not d["knn_trivial"]:
        best_model = min(v for v in (d["gnn"], d["gem"]) if v is not None)
        win = "kNN WINS" if d["knn"] < best_model else ""
    print(f'{d["label"]:24s} {g:>7s} {sp:>5s} {m:>7s} {k:>8s}  {win}')
print("wrote", OUT)
