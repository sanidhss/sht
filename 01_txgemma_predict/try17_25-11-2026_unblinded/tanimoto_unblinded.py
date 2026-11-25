"""Tanimoto 5-NN, BLINDED vs UNBLINDED, plus a random-chance baseline, on pool_v1.

Three arms on the identical frozen 1,933-drug pool:

  blinded    actives = train_blind.jsonl, reference held out (try16's number, recomputed
             here so all three arms come from one code path)
  unblinded  actives = train_blind.jsonl PLUS the reference drug with its own measured
             label. The 5-NN then finds the reference at Tanimoto 1.000 and reads its
             own answer back, so this is a CEILING, not a result: it measures how much
             of the task is pure retrieval.
  random     a uniformly random ranking. Reported analytically (expected rank
             (n+1)/2 = 967, expected percentile 50%) and verified empirically, which
             also checks that rank_in_pool() is unbiased.

The unblinded arm is the honest upper bound for any lookup-style method, and the gap
between it and the blinded arm is the part of the problem that is actually prediction
rather than recall.

CPU only. No GPU, no network, no model weights.
"""

import json
import os
import random
import statistics
import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_common as pc

KNN_K = int(os.environ.get("KNN_K", "5"))
TRIVIAL_AT = 0.95
N_RANDOM_TRIALS = int(os.environ.get("N_RANDOM_TRIALS", "20000"))
OUT_ROOT = os.path.join(pc.ORPHAN, "try17_unblinded")
MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smiles):
    m = Chem.MolFromSmiles(smiles)
    return MFPGEN.GetFingerprint(m) if m is not None else None


def knn_scores(cand_fps, active_fps, active_labels, k):
    """Similarity-weighted mean label of the k nearest actives. Lower = more potent.
    Verbatim from 05_multi_agent/prefilter_recall.py."""
    scores = []
    for cf in cand_fps:
        sims = DataStructs.BulkTanimotoSimilarity(cf, active_fps)
        idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
        wsum = sum(sims[i] for i in idx)
        if wsum <= 0:
            scores.append(float("inf"))
        else:
            scores.append(sum(sims[i] * active_labels[i] for i in idx) / wsum)
    return scores


def random_chance(n_pool, trials, seed=0):
    """Empirical distribution of a uniformly random ranking. The analytic answer is
    known -- rank is uniform on 1..n, so percentile is uniform on (0, 100] with mean 50
    -- and this confirms the rank machinery reproduces it rather than being biased by
    the reference's position in the pool list."""
    rng = random.Random(seed)
    pcts = []
    for _ in range(trials):
        ref = rng.random()
        others = [rng.random() for _ in range(n_pool - 1)]
        better = sum(1 for v in others if v < ref)
        pcts.append(100.0 * (better + 1) / n_pool)
    pcts.sort()
    def q(p):
        return pcts[min(len(pcts) - 1, int(p * len(pcts)))]
    return {"trials": trials, "mean_percentile": round(statistics.mean(pcts), 3),
            "expected_rank": (n_pool + 1) / 2,
            "p05": round(q(0.05), 2), "p25": round(q(0.25), 2),
            "p50": round(q(0.50), 2), "p75": round(q(0.75), 2),
            "p95": round(q(0.95), 2)}


def main():
    pool = pc.verify_pool()
    structures = pc.pool_structures(pool)
    pool_smis = [c for c, _ in structures]
    pool_fps = [fp(c) for c in pool_smis]
    keep = [i for i, f in enumerate(pool_fps) if f is not None]
    pool_smis = [pool_smis[i] for i in keep]
    pool_fps = [pool_fps[i] for i in keep]
    n_pool = len(pool_smis)
    print(f"pool fingerprinted: {n_pool} compounds\n", flush=True)

    rand = random_chance(n_pool, N_RANDOM_TRIALS)
    print(f"RANDOM CHANCE over {rand['trials']} trials: mean percentile "
          f"{rand['mean_percentile']}%, expected rank {rand['expected_rank']:.0f}, "
          f"5th pct {rand['p05']}%, 95th pct {rand['p95']}%\n", flush=True)

    results = {}
    print(f"{'target':13}{'drug':12}{'blinded':>22}{'unblinded':>22}  leak")
    for target in pc.ALL_TARGETS:
        ref = pc.reference_of(target)
        actives = pc.load_actives(target)
        leaked_already = ref["smiles"] in actives

        in_pool = ref["smiles"] in set(pool_smis)
        row = {"target": target, "reference": ref["name"],
               "reference_real_label": ref["real"],
               "reference_in_pool": in_pool,
               "reference_already_in_actives": leaked_already,
               "n_actives_blinded": len(actives)}

        def arm(active_map):
            smis = list(active_map)
            labels = [active_map[s] for s in smis]
            fps = [fp(s) for s in smis]
            ok = [i for i, f in enumerate(fps) if f is not None]
            fps = [fps[i] for i in ok]
            labels = [labels[i] for i in ok]
            rfp = fp(ref["smiles"])
            sims = DataStructs.BulkTanimotoSimilarity(rfp, fps) if fps else []
            max_sim = max(sims) if sims else 0.0
            out = {"n_actives": len(fps), "max_sim_to_actives": round(max_sim, 4),
                   "trivial_flag": max_sim >= TRIVIAL_AT, "rank": None,
                   "percentile": None}
            if in_pool:
                scores = knn_scores(pool_fps, fps, labels, KNN_K)
                rank, tied = pc.rank_in_pool(scores[pool_smis.index(ref["smiles"])],
                                             scores)
                out["rank"] = rank
                out["percentile"] = round(100.0 * rank / n_pool, 3)
                out["n_tied_with_ref"] = tied
            return out

        row["blinded"] = arm(dict(actives))

        unb = dict(actives)
        unb[ref["smiles"]] = float(ref["real"])
        row["unblinded"] = arm(unb)
        row["random"] = rand

        results[target] = row
        b, u = row["blinded"], row["unblinded"]
        bs = f"rank {b['rank']}/{n_pool} ({b['percentile']:.2f}%)" if in_pool else "NOT IN POOL"
        us = f"rank {u['rank']}/{n_pool} ({u['percentile']:.2f}%)" if in_pool else "NOT IN POOL"
        flag = " TRIVIAL(blind)" if b["trivial_flag"] else ""
        print(f"{target:13}{ref['name']:12}{bs:>22}{us:>22}  "
              f"max_sim {b['max_sim_to_actives']:.3f}{flag}", flush=True)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out = os.path.join(OUT_ROOT, "tanimoto_blind_vs_unblind.json")
    with open(out, "w") as fh:
        json.dump({"knn_k": KNN_K, "n_pool": n_pool, "random_chance": rand,
                   "results": results}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
