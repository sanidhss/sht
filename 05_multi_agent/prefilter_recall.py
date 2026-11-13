"""
How deep does a chemistry pre-filter have to go to catch the reference drug?

This decides `k` for the multi-agent protocol, which the paper leaves unspecified.

Why it matters here and not in the paper: RareAgent's candidate space is 74 drugs, so
debating "the top k" is close to debating everything. This project's library is 3,420
approved drugs -- about 46x larger. Something has to cut 3,420 down to k before any
debate happens, and if the reference drug is not in that top-k, no amount of debate
recovers it. So the pre-filter's recall@k is an upper bound on the whole arm, and k
should be chosen from this curve rather than from what happens to fit in context.

Pre-filter measured: Tanimoto k-NN over Morgan-2 2048-bit fingerprints, predicting a
compound's potency as the similarity-weighted mean of its `KNN_K` nearest neighbours
among compounds with MEASURED activity against that target. This is the same family as
`01_txgemma_predict`'s `descriptor_baseline_predict.py` -- the baseline that already
beats the fine-tuned 9B on 5 of 6 targets and the 27B chat arm on 4 of 4 folds. It is
the strongest cheap signal available, so it is the fair thing to size k against.

Blinding: actives come from `train_blind.jsonl`, which already has the reference drug
held out by `prep_orphan_target_data.py`. The script asserts this rather than trusting
it.

Leak control (the number that decides whether any of this means anything): for every
target it reports the maximum Tanimoto similarity between the reference drug and the
actives pool. The reference is held out, but a near-identical analog may not be -- the
try09 scaffold holdout happens at train time, not in this file. If max similarity is
~1.0 the recall is trivial and must not be reported as a result.

CPU only. RDKit, no GPU, no network, no model weights.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ORPHAN = Path("/work/hdd/bisa/sshah21/results/txgemma_lora_orphan")
LIBRARY_JSON = Path("/work/hdd/bisa/sshah21/results/reasoning_screen/chembl_approved.json")
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))

KNN_K = int(os.environ.get("KNN_K", "5"))
K_GRID = [10, 20, 50, 100, 200, 500, 1000, 2000]

SMILES_RE = re.compile(r"Drug SMILES: (\S+)")

MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canon(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def fp(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    return MFPGEN.GetFingerprint(m) if m is not None else None


def load_actives(target: str) -> dict[str, float]:
    """{canonical smiles: label}. The 5x SMILES augmentation is collapsed by canonical
    form, so each real compound is counted once."""
    out: dict[str, float] = {}
    path = ORPHAN / target / "train_blind.jsonl"
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            m = SMILES_RE.search(rec["prompt"])
            if not m:
                continue
            c = canon(m.group(1))
            if c is None:
                continue
            out.setdefault(c, float(rec["answer"]))
    return out


def load_library() -> list[dict[str, str]]:
    data = json.loads(LIBRARY_JSON.read_text())
    recs = data if isinstance(data, list) else list(data.values())
    out = []
    for r in recs:
        smi = r.get("canonical_smiles") or r.get("smiles")
        if not smi:
            continue
        c = canon(smi)
        if c is None:
            continue
        out.append({"name": (r.get("name") or "").upper(), "chembl_id": r.get("chembl_id", ""),
                    "smiles": c})
    return out


def knn_scores(lib_fps, active_fps, active_labels, k: int) -> list[float]:
    """Similarity-weighted mean label of the k nearest actives. Lower = more potent."""
    scores = []
    for lf in lib_fps:
        sims = DataStructs.BulkTanimotoSimilarity(lf, active_fps)
        idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
        wsum = sum(sims[i] for i in idx)
        if wsum <= 0:
            scores.append(float("inf"))  # nothing similar: unrankable, sorts last
        else:
            scores.append(sum(sims[i] * active_labels[i] for i in idx) / wsum)
    return scores


def run_target(target: str, library: list[dict[str, str]]) -> dict | None:
    eval_path = ORPHAN / target / "eval_set.json"
    if not eval_path.exists():
        return None
    ref = json.loads(eval_path.read_text())["reference"]
    ref_smi = canon(ref["smiles"])
    ref_name = (ref["name"] or "").upper()

    actives = load_actives(target)

    # Blinding assertion, not assumption.
    leaked = ref_smi in actives
    if leaked:
        print(f"  !! {target}: reference IS in the actives pool -- blinding broken")

    active_smis = list(actives)
    active_labels = [actives[s] for s in active_smis]
    active_fps = [fp(s) for s in active_smis]
    keep = [i for i, f in enumerate(active_fps) if f is not None]
    active_fps = [active_fps[i] for i in keep]
    active_labels = [active_labels[i] for i in keep]

    # Leak control: how close is the nearest active to the reference?
    ref_fp = fp(ref_smi)
    sims_to_ref = DataStructs.BulkTanimotoSimilarity(ref_fp, active_fps)
    max_sim = max(sims_to_ref) if sims_to_ref else 0.0
    n_above_95 = sum(1 for s in sims_to_ref if s >= 0.95)
    n_above_80 = sum(1 for s in sims_to_ref if s >= 0.80)

    # Score the library. The reference is spiked in if the approved library does not
    # carry it (02_txgemma_chat has to do this for ambroxol, which is not max_phase=4).
    pool = list(library)
    present = any(r["smiles"] == ref_smi for r in pool)
    if not present:
        pool.append({"name": ref_name, "chembl_id": "SPIKED", "smiles": ref_smi})

    lib_fps = [fp(r["smiles"]) for r in pool]
    ok = [i for i, f in enumerate(lib_fps) if f is not None]
    pool = [pool[i] for i in ok]
    lib_fps = [lib_fps[i] for i in ok]

    scores = knn_scores(lib_fps, active_fps, active_labels, KNN_K)
    order = sorted(range(len(pool)), key=lambda i: scores[i])
    rank = next(
        (r for r, i in enumerate(order, 1) if pool[i]["smiles"] == ref_smi), None
    )

    return {
        "target": target,
        "reference": ref["name"],
        "reference_in_library": present,
        "n_actives": len(active_fps),
        "pool_size": len(pool),
        "rank": rank,
        "percentile": (100.0 * rank / len(pool)) if rank else None,
        "max_tanimoto_to_actives": round(max_sim, 4),
        "n_actives_sim_ge_0.95": n_above_95,
        "n_actives_sim_ge_0.80": n_above_80,
        "reference_leaked_into_actives": leaked,
    }


def main() -> None:
    library = load_library()
    targets = sorted(
        d.name for d in ORPHAN.iterdir()
        if (d / "eval_set.json").exists() and (d / "train_blind.jsonl").exists()
    )

    print("=" * 94)
    print("PRE-FILTER RECALL@K — how deep must a chemistry filter go to catch the reference?")
    print("=" * 94)
    print(f"library           : {len(library)} approved compounds")
    print(f"pre-filter        : Tanimoto {KNN_K}-NN, Morgan-2 2048-bit, "
          f"similarity-weighted mean label")
    print(f"targets available : {len(targets)} ({', '.join(targets)})")
    print()
    print("NOTE: the 12 pins include 4 cases with too few measured activities to support")
    print("any chemistry pre-filter at all (PKU 2 records, propionic acidemia 4, MH 10,")
    print("HLCS 11). They have no prepared target dir and are absent below -- that")
    print("absence is itself the result for those cells.")
    print()

    results = []
    for t in targets:
        print(f"[{t}]")
        r = run_target(t, library)
        if r:
            results.append(r)
            print(f"  rank {r['rank']}/{r['pool_size']}  "
                  f"({r['percentile']:.2f}%)  actives={r['n_actives']}  "
                  f"maxTanimoto={r['max_tanimoto_to_actives']}")

    print()
    print("=" * 94)
    print("RESULTS")
    print("=" * 94)
    print(f"{'target':<14}{'reference':<16}{'rank':>8}{'pct':>8}{'actives':>9}"
          f"{'maxSim':>8}{'>=.95':>7}{'>=.80':>7}  leak?")
    print("-" * 94)
    for r in sorted(results, key=lambda x: x["rank"] or 10**9):
        flag = "TRIVIAL" if r["max_tanimoto_to_actives"] >= 0.95 else (
            "close" if r["max_tanimoto_to_actives"] >= 0.80 else "")
        print(f"{r['target']:<14}{r['reference'][:15]:<16}{str(r['rank']):>8}"
              f"{r['percentile']:>7.2f}%{r['n_actives']:>9}"
              f"{r['max_tanimoto_to_actives']:>8.3f}{r['n_actives_sim_ge_0.95']:>7}"
              f"{r['n_actives_sim_ge_0.80']:>7}  {flag}")

    print()
    print("RECALL@K  (fraction of targets whose reference is within the top k)")
    print("-" * 94)
    n = len(results)
    print(f"{'k':>8}  {'recall':>8}  {'n':>5}   targets caught")
    for k in K_GRID:
        caught = [r["target"] for r in results if r["rank"] and r["rank"] <= k]
        print(f"{k:>8}  {len(caught)/n:>7.0%}  {len(caught):>3}/{n}   "
              f"{', '.join(sorted(caught)) if caught else '-'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "prefilter_recall.json"
    out.write_text(json.dumps(
        {"knn_k": KNN_K, "library_size": len(library), "results": results,
         "recall_at_k": {k: sum(1 for r in results if r["rank"] and r["rank"] <= k) / n
                         for k in K_GRID}},
        indent=2))
    print(f"\nwrote {out.resolve()}")


if __name__ == "__main__":
    main()
