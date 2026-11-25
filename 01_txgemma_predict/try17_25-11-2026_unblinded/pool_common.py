"""Shared pool loading + verification for try16.

Copied-in constants rather than imported from 06_equalized_pool: a try folder snapshots
the exact code that ran, and an import would let a later edit elsewhere silently change
what this run means. The sha256 check is the contract that makes that safe -- if the
pool file is ever rebuilt, this script STOPS rather than scoring a different pool and
reporting it under the same name.
"""

import hashlib
import json
import os
import re
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

POOL_DIR = "/work/hdd/bisa/sshah21/results/equalized_pool"
POOL_JSON = os.path.join(POOL_DIR, "pool_v1.json")
POOL_SHA = os.path.join(POOL_DIR, "pool_v1.sha256")
ORPHAN = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
OUT_ROOT = os.path.join(ORPHAN, "try16_equalized_pool")

ALL_TARGETS = ["gaucher", "cf", "fabry", "attr",
               "cel", "mastocytosis", "nsiad", "tyrosinemia"]

SMILES_RE = re.compile(r"Drug SMILES: (\S+)")


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def verify_pool():
    """STEP 0. Returns the pool records. Exits non-zero on any mismatch -- a silently
    different pool would invalidate every cross-architecture comparison this file
    exists to enable."""
    with open(POOL_SHA) as fh:
        expected = fh.read().split()[0].strip()
    h = hashlib.sha256()
    with open(POOL_JSON, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        print(f"FATAL: pool sha256 mismatch\n  expected {expected}\n  actual   {actual}",
              flush=True)
        sys.exit(2)
    pool = json.load(open(POOL_JSON))
    print(f"pool verified: {len(pool)} records, sha256 {actual}", flush=True)
    return pool


def pool_structures(pool):
    """[(canonical_smiles, record)] in file order, de-duplicated by canonical form.
    pool_v1 has no duplicate structures (verified: 1933 records -> 1933 canonical
    forms), but de-duplicating here means a future pool version cannot silently give
    one compound two chances to outrank the reference."""
    seen, out = set(), []
    for r in pool:
        c = canon(r["canonical_smiles"])
        if c is None or c in seen:
            continue
        seen.add(c)
        out.append((c, r))
    return out


def reference_of(target):
    """The held-out reference drug for a target, from its frozen eval set."""
    ref = json.load(open(os.path.join(ORPHAN, target, "eval_set.json")))["reference"]
    return {"name": ref["name"], "smiles": canon(ref["smiles"]),
            "prompt": ref["prompt"], "real": ref["score"]}


def load_actives(target):
    """{canonical smiles: label} from train_blind.jsonl. The 5x SMILES augmentation is
    collapsed by canonical form, so each real compound counts once. Same loader as
    05_multi_agent/prefilter_recall.py."""
    out = {}
    with open(os.path.join(ORPHAN, target, "train_blind.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            m = SMILES_RE.search(rec["prompt"])
            if not m:
                continue
            c = canon(m.group(1))
            if c is not None:
                out.setdefault(c, float(rec["answer"]))
    return out


def rank_in_pool(ref_score, all_scores):
    """Rank of the reference among pool scores, LOWER score = more potent = better.

    Returns best-case rank (1 + strictly-better) and the number of compounds tied with
    the reference. Both are reported because the greedy readout collapses hard -- try14
    measured 54 distinct greedy values across 500 decoys -- so a bare rank would flatter
    greedy enormously by breaking every tie in the reference's favour. digit_exp
    effectively never ties.
    """
    if ref_score is None:
        return None, None
    vals = [v for v in all_scores if v is not None]
    if not vals:
        return None, None
    better = sum(1 for v in vals if v < ref_score)
    tied = sum(1 for v in vals if v == ref_score)
    return better + 1, tied
