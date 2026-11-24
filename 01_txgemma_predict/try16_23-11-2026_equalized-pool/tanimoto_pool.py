"""STEP 2 -- Tanimoto 5-NN baseline over the frozen equalized pool.

The same estimator as 05_multi_agent/prefilter_recall.py: a compound's predicted
potency is the similarity-weighted mean label of its KNN_K nearest neighbours among
compounds with MEASURED activity against that target (Morgan-2, 2048-bit). Only the
candidate set changes -- the 1,933-drug equalized pool instead of the 3,417-drug
library -- so this arm and the adapter arm rank the same compounds and their ranks are
directly comparable.

Blinding is asserted, not assumed: the reference must not appear among the actives.

Leak control, which is the number that decides whether any of this means anything:
the reference is held out, but a near-identical analogue may not be. Max Tanimoto from
the reference to any active is reported for every target, and >= 0.95 is flagged
TRIVIAL. That control is what exposed nsiad and fabry, so it is computed for all eight
targets even though only six can be ranked on this pool.

CPU only. RDKit, no GPU, no network, no model weights.
"""

import json
import os
import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_common as pc

KNN_K = int(os.environ.get("KNN_K", "5"))
TRIVIAL_AT = 0.95
MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smiles):
    m = Chem.MolFromSmiles(smiles)
    return MFPGEN.GetFingerprint(m) if m is not None else None


def knn_scores(cand_fps, active_fps, active_labels, k):
    """Similarity-weighted mean label of the k nearest actives. Lower = more potent.
    Verbatim from prefilter_recall.py."""
    scores = []
    for cf in cand_fps:
        sims = DataStructs.BulkTanimotoSimilarity(cf, active_fps)
        idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
        wsum = sum(sims[i] for i in idx)
        if wsum <= 0:
            scores.append(float("inf"))  # nothing similar: unrankable, sorts last
        else:
            scores.append(sum(sims[i] * active_labels[i] for i in idx) / wsum)
    return scores


def main():
    pool = pc.verify_pool()
    structures = pc.pool_structures(pool)
    pool_smis = [c for c, _ in structures]
    pool_fps = [fp(c) for c in pool_smis]
    bad = [i for i, f in enumerate(pool_fps) if f is None]
    if bad:
        print(f"  {len(bad)} pool compounds failed fingerprinting and are dropped")
    keep = [i for i, f in enumerate(pool_fps) if f is not None]
    pool_smis = [pool_smis[i] for i in keep]
    pool_fps = [pool_fps[i] for i in keep]
    print(f"pool fingerprinted: {len(pool_fps)} compounds\n", flush=True)

    os.makedirs(pc.OUT_ROOT, exist_ok=True)
    results = {}

    for target in pc.ALL_TARGETS:
        ref = pc.reference_of(target)
        actives = pc.load_actives(target)

        # Blinding assertion, not assumption.
        leaked = ref["smiles"] in actives
        if leaked:
            print(f"!! {target}: reference IS in the actives pool -- blinding broken")

        smis = list(actives)
        labels = [actives[s] for s in smis]
        fps = [fp(s) for s in smis]
        ok = [i for i, f in enumerate(fps) if f is not None]
        fps = [fps[i] for i in ok]
        labels = [labels[i] for i in ok]

        # Leak control.
        rfp = fp(ref["smiles"])
        sims = DataStructs.BulkTanimotoSimilarity(rfp, fps) if fps else []
        max_sim = max(sims) if sims else 0.0
        trivial = max_sim >= TRIVIAL_AT

        in_pool = ref["smiles"] in set(pool_smis)
        rank = pct = None
        if in_pool:
            scores = knn_scores(pool_fps, fps, labels, KNN_K)
            ref_score = scores[pool_smis.index(ref["smiles"])]
            rank, tied = pc.rank_in_pool(ref_score, scores)
            pct = 100.0 * rank / len(pool_smis)

        results[target] = {
            "target": target, "reference": ref["name"],
            "reference_in_pool": in_pool,
            "n_actives": len(fps),
            "n_pool": len(pool_smis),
            "rank": rank,
            "percentile": round(pct, 3) if pct is not None else None,
            "max_sim_to_actives": round(max_sim, 4),
            "n_actives_sim_ge_0.95": sum(1 for s in sims if s >= 0.95),
            "n_actives_sim_ge_0.80": sum(1 for s in sims if s >= 0.80),
            "trivial_flag": trivial,
            "reference_leaked_into_actives": leaked,
        }
        flag = "  *** TRIVIAL ***" if trivial else ""
        loc = f"rank {rank}/{len(pool_smis)} ({pct:.2f}%)" if in_pool else "NOT IN POOL"
        print(f"{target:13} {ref['name']:12} {loc:28} "
              f"max_sim {max_sim:.3f} actives {len(fps)}{flag}", flush=True)

    out = os.path.join(pc.OUT_ROOT, "tanimoto_pool_v1.json")
    with open(out, "w") as fh:
        json.dump({"knn_k": KNN_K, "trivial_at": TRIVIAL_AT, "results": results},
                  fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
