"""
Non-LLM baseline for the NUMERIC PREDICT task -- the number the fine-tune has to clear.

WHY THIS FILE EXISTS INSTEAD OF REUSING 02_txgemma_chat/try11/descriptor_baseline.py
------------------------------------------------------------------------------------
That script cannot be run on a new target, and its numbers are not in these units. It:
  - reads `reasoning_chains/<fold>.json`, which exists only for the four chat folds
    (fabry, gaucher, cf, attr) -- there is no chain for cel/mastocytosis/nsiad/tyrosinemia;
  - carries a HAND-WRITTEN RDKit filter per fold, translated from that fold's
    `necessary_condition` TEXT. For a new target there is no condition text to translate,
    and writing a filter from the reference drug's own structure would be circular -- that
    script's own docstring forbids exactly that ("written from the condition text alone and
    never from the true drug's structure"), and its README notes the filters were written
    after seeing which folds failed;
  - measures RANK IN THE 3,417-DRUG APPROVED LIBRARY for a text-screening task. The predict
    track measures ENRICHMENT PERCENTILE AGAINST 500 CROSS-TARGET DECOYS. The chat track's
    "2x to 48x better than the model" is a real result in its own units, but it is not a
    number this track's percentiles can be compared against.

So this is the same IDEA -- "can a cheap mechanical method on the same data beat the
9B-parameter fine-tune?" -- rebuilt to be (a) target-agnostic, so it works on targets
nobody has hand-written a filter for, and (b) reported in the IDENTICAL units as the
model, against the same decoys, from the same training compounds, under the same
scaffold holdout.

THE TWO BASELINES
-----------------
Both are fit on exactly the compounds the LLM trains on (same scaffold holdout of the
hard negatives) and scored on exactly what the LLM is scored on (reference + 3 hard
negatives + the same 500 MaxMin decoys, same builder and seed).

  1. knn  -- Tanimoto-weighted k-nearest-neighbour regression on Morgan fingerprints
             (radius 2, 2048 bits), k=5. The standard cheap chemoinformatics baseline:
             "this compound is probably about as potent as the training compounds it
             most resembles." If a 9B model fine-tuned on these compounds cannot beat
             nearest-neighbour lookup, it has learned nothing a lookup table doesn't.

  2. ridge -- least-squares linear regression on 14 interpretable RDKit descriptors
             (MW, logP, TPSA, rotatable bonds, H-bond donors/acceptors, ring counts,
             aromatic rings, heavy atoms, fraction Csp3, formal charge, stereocentres,
             heteroatoms) with an intercept and L2 regularization. This is the
             "descriptor baseline" in the literal sense: no structural lookup at all,
             just bulk physicochemistry.

Ridge is solved with numpy.linalg.lstsq on the augmented normal equations -- scikit-learn
is not installed in this environment and is not needed for a 14-feature fit.

Neither baseline sees the reference drug's label. Neither uses a GPU. Whole run is
seconds per target.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

RDLogger.DisableLog("rdApp.*")

DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
TRY14_OUT = os.path.join(DATA_DIR, "try14_new_targets")
ALL_TARGETS = ["gaucher", "cf", "fabry", "attr", "cel", "mastocytosis", "nsiad", "tyrosinemia"]
N_DECOYS = 500
K_NEIGHBOURS = 5
RIDGE_LAMBDA = 1.0


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def scaffold_of(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


DESCRIPTORS = [
    ("mw", Descriptors.MolWt), ("logp", Descriptors.MolLogP),
    ("tpsa", rdMolDescriptors.CalcTPSA),
    ("rotb", rdMolDescriptors.CalcNumRotatableBonds),
    ("hbd", rdMolDescriptors.CalcNumHBD), ("hba", rdMolDescriptors.CalcNumHBA),
    ("rings", rdMolDescriptors.CalcNumRings),
    ("arom", rdMolDescriptors.CalcNumAromaticRings),
    ("heavy", lambda m: m.GetNumHeavyAtoms()),
    ("fcsp3", rdMolDescriptors.CalcFractionCSP3),
    ("charge", Chem.GetFormalCharge),
    ("stereo", lambda m: len(Chem.FindMolChiralCenters(
        m, includeUnassigned=True, useLegacyImplementation=False))),
    ("hetero", lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() not in ("C", "H"))),
    ("nO", lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() == "O")),
]


def desc_vector(mol):
    out = []
    for _, f in DESCRIPTORS:
        try:
            out.append(float(f(mol)))
        except Exception:
            out.append(0.0)
    return out


def load_grouped(target):
    """compound -> label, from the same train_blind.jsonl the LLM trains on."""
    groups = defaultdict(list)
    for line in open(os.path.join(DATA_DIR, target, "train_blind.jsonl")):
        d = json.loads(line)
        m = re.search(r"Drug SMILES: (.+)", d["prompt"])
        if not m:
            continue
        c = canon(m.group(1).strip())
        if c:
            groups[c].append(float(d["answer"]))
    return {c: sum(v) / len(v) for c, v in groups.items()}


def build_decoys(target, n):
    """Identical to the eval harnesses': same pool rule, same MaxMin, same seed=0."""
    own = set(load_grouped(target).keys())
    pool = set()
    for t in ALL_TARGETS:
        if t == target or not os.path.exists(os.path.join(DATA_DIR, t, "train_blind.jsonl")):
            continue
        pool |= set(load_grouped(t).keys())
    pool = sorted(pool - own)
    mols = [(s, Chem.MolFromSmiles(s)) for s in pool]
    mols = [(s, m) for s, m in mols if m is not None]
    fps = [fp(m) for _, m in mols]
    k = min(n, len(fps))
    idx = list(MaxMinPicker().LazyBitVectorPick(fps, len(fps), k, seed=0))
    return [mols[i][0] for i in idx]


def fit_ridge(X, y, lam=RIDGE_LAMBDA):
    """Standardize, then closed-form ridge via lstsq. numpy only -- no sklearn here."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    A = np.vstack([Xs, np.sqrt(lam) * np.eye(Xs.shape[1])])
    b = np.concatenate([y, np.zeros(Xs.shape[1])])
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return lambda Xn: (np.hstack([(Xn - mu) / sd, np.ones((len(Xn), 1))]) @ w)


def knn_predict(train_fps, train_y, query_fps, k=K_NEIGHBOURS):
    out = []
    for q in query_fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(q, train_fps))
        top = np.argsort(-sims)[:k]
        w = sims[top]
        # All-zero similarity (nothing in training resembles it at all) -> fall back to
        # the training mean rather than dividing by zero.
        out.append(float(np.average(train_y[top], weights=w)) if w.sum() > 0
                   else float(train_y.mean()))
    return out


def enrichment(ref, vals):
    vv = [v for v in vals if v is not None]
    if ref is None or not vv:
        return None
    better = sum(1 for v in vv if v < ref)
    return {"rank": better, "n": len(vv), "percentile": 100.0 * better / len(vv)}


def run(target):
    labels = load_grouped(target)
    eval_set = json.load(open(os.path.join(DATA_DIR, target, "eval_set.json")))
    ref_smiles = canon(eval_set["reference"]["smiles"])
    hard = [{"smiles": canon(c["smiles"]), "real": c["score"],
             "name": c.get("name") or c["smiles"][:20]} for c in eval_set["hard_negatives"]]

    # SAME scaffold holdout as the LLM harnesses: remove the hard negatives and anything
    # sharing their skeleton, so the baseline is not quietly fit on the answers.
    hard_scaf = {scaffold_of(h["smiles"]) for h in hard}
    hard_scaf.discard(None)
    train = {c: y for c, y in labels.items() if scaffold_of(c) not in hard_scaf}
    print(f"[{target}] {len(labels)} compounds -> {len(train)} after the same scaffold "
          f"holdout the LLM gets ({len(labels) - len(train)} removed)")
    if len(train) < 10:
        print(f"[{target}] too few training compounds to fit a baseline")
        return None

    tr_smiles = sorted(train)
    tr_mols = [Chem.MolFromSmiles(c) for c in tr_smiles]
    keep = [i for i, m in enumerate(tr_mols) if m is not None]
    tr_smiles = [tr_smiles[i] for i in keep]
    tr_mols = [tr_mols[i] for i in keep]
    tr_y = np.array([train[c] for c in tr_smiles], dtype=float)
    tr_fps = [fp(m) for m in tr_mols]
    tr_X = np.array([desc_vector(m) for m in tr_mols], dtype=float)

    decoys = build_decoys(target, N_DECOYS)
    queries = [ref_smiles] + [h["smiles"] for h in hard] + decoys
    q_mols = [Chem.MolFromSmiles(s) for s in queries]
    ok = [i for i, m in enumerate(q_mols) if m is not None]
    q_mols = [q_mols[i] for i in ok]
    q_names = [queries[i] for i in ok]
    n_ref_hard = sum(1 for i in ok if i <= len(hard))

    preds = {}
    preds["knn"] = knn_predict(tr_fps, tr_y, [fp(m) for m in q_mols])
    ridge = fit_ridge(tr_X, tr_y)
    preds["ridge"] = list(ridge(np.array([desc_vector(m) for m in q_mols], dtype=float)))

    out = {"target": target, "n_train": len(tr_smiles), "n_decoys": len(q_mols) - n_ref_hard,
           "reference_real": eval_set["reference"]["score"]}
    for method, p in preds.items():
        ref_v = p[0]
        hard_v = p[1:n_ref_hard]
        dec_v = p[n_ref_hard:]
        e = enrichment(ref_v, dec_v)
        beats = all(ref_v < h for h in hard_v) if hard_v else None
        out[method] = {"reference": ref_v, "hard": hard_v, "beats_hard": beats,
                       "enrichment": e}
        print(f"[{target}] {method:6s} ref={ref_v:7.1f} (real {out['reference_real']}) | "
              f"beats_hard={beats} | enrichment {e['percentile']:.1f} percentile "
              f"({e['rank']}/{e['n']})")
    return out


if __name__ == "__main__":
    want = sys.argv[1:] or [t for t in ALL_TARGETS
                            if os.path.exists(os.path.join(DATA_DIR, t, "eval_set.json"))]
    os.makedirs(TRY14_OUT, exist_ok=True)
    results = {}
    for t in want:
        if not os.path.exists(os.path.join(DATA_DIR, t, "eval_set.json")):
            print(f"[{t}] no eval_set.json -- not prepped, skipping")
            continue
        try:
            r = run(t)
            if r:
                results[t] = r
        except Exception as e:
            print(f"[{t}] baseline FAILED: {type(e).__name__}: {e}")
    path = os.path.join(TRY14_OUT, "descriptor_baseline.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {path}")
    print(f"\n{'target':14s} {'n_train':>7s} {'knn %ile':>9s} {'knn beats':>10s} "
          f"{'ridge %ile':>11s} {'ridge beats':>12s}")
    for t, r in results.items():
        print(f"{t:14s} {r['n_train']:>7} "
              f"{r['knn']['enrichment']['percentile']:>9.1f} "
              f"{str(r['knn']['beats_hard']):>10s} "
              f"{r['ridge']['enrichment']['percentile']:>11.1f} "
              f"{str(r['ridge']['beats_hard']):>12s}")
