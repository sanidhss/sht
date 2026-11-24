"""
Restrict try11's saved screens to the frozen equalized pool (pool_v1) and re-rank.

No model calls: try11/try10 persisted full per-compound margins with chembl_id, so this
is a join. Ranks here are NOT comparable to the 3,417-compound numbers -- that is the
point, they are being put on the shared ruler used by the other architecture arms.

Carries the register comparison forward: fabry was screened under both a physicochemical
and a pharmacological phrasing of the same condition, and gaucher's authored condition is
in that same pharmacological register. Both pairs are re-ranked so the finding survives.
"""
import csv, hashlib, json, os, sys

POOL = "/work/hdd/bisa/sshah21/results/equalized_pool/pool_v1.json"
SHA  = "/work/hdd/bisa/sshah21/results/equalized_pool/pool_v1.sha256"
SCR  = "/work/hdd/bisa/sshah21/results/reasoning_screen"
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_pool_v1.csv")

# (target, register, saved screen file)
ARMS = [
    ("fabry",   "physchem",        "screen_fabry_fabryphys2_base.json"),
    ("fabry",   "pharmacological", "screen_fabry_fabrypharm_base.json"),
    ("gaucher", "physchem",        "screen_gaucher_physchem_base.json"),
    ("gaucher", "pharmacological", "screen_gaucher_authored_base.json"),
    ("cf",      "physchem",        "screen_cf_physchem_base.json"),
    ("attr",    "physchem",        "screen_attr_physchem_base.json"),
]

# STEP 0 -- verify the pool before anything else.
expected = open(SHA).read().split()[0]
actual = hashlib.sha256(open(POOL, "rb").read()).hexdigest()
if expected != actual:
    sys.exit(f"STOP: pool_v1.json sha256 mismatch\n  expected {expected}\n  actual   {actual}")
pool = json.load(open(POOL))
pool_ids = {r["chembl_id"] for r in pool if r.get("chembl_id")}
print(f"pool_v1 verified: {actual[:16]}... | {len(pool)} rows, {len(pool_ids)} unique chembl_id")

rows = []
for target, register, fname in ARMS:
    d = json.load(open(os.path.join(SCR, fname)))
    ranked = d["ranked"]
    ref = d["true_drug"]
    # Original full-screen rank, for reference only.
    full = sorted(ranked, key=lambda r: -r["margin"])
    rank_full = next((i for i, r in enumerate(full, 1)
                      if (r["name"] or "").upper() == ref.upper()), None)

    # STEP 1 -- join on chembl_id. Spiked analogs and spiked reference drugs carry
    # chembl_id=None and are correctly dropped: they are not in the equalized pool.
    sub = [r for r in ranked if r.get("chembl_id") in pool_ids]
    sub.sort(key=lambda r: -r["margin"])
    rank_pool = next((i for i, r in enumerate(sub, 1)
                      if (r["name"] or "").upper() == ref.upper()), None)
    n_pool = len(sub)
    pct = round(100.0 * rank_pool / n_pool, 2) if rank_pool else None
    rows.append(dict(target=target, ref_drug=ref, register=register,
                     rank_pool=rank_pool if rank_pool else "NOT_IN_POOL",
                     n_pool=n_pool, percentile_pool=pct if pct is not None else "",
                     rank_3417=rank_full))
    print(f"{target:8s} {register:16s} pool {str(rank_pool):>6s}/{n_pool:<5d} "
          f"({pct if pct is not None else 'n/a'}%)   full-screen rank {rank_full}")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["target", "ref_drug", "register", "rank_pool",
                                      "n_pool", "percentile_pool", "rank_3417"])
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {OUT}")
