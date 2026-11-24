# Try 12 — try11's screen re-ranked on the frozen equalized pool (pool_v1)

**Date:** 23-11-2026
**Script:** `rerank_on_pool.py` (no GPU, no model calls)
**Output:** `results_pool_v1.csv`

This arm is **descoped from the architecture comparison**. The only purpose is to put its
numbers on the same ruler as the other arms and to carry the register finding forward.
It is not a competing arm.

## Step 0 — pool verified

`pool_v1.json` sha256 `9d89c3ce758b7ec7c8bea2866f6448a9ea71b372f0f3c3f92c8c6b690653e308`
matches `pool_v1.sha256`. 1,933 rows, 1,933 unique `chembl_id`.

## Step 1 — subset, no re-screening needed

try10/try11 persisted full per-compound margins with `chembl_id`, so this is a join, not
a re-run. Compounds are matched on `chembl_id`; the spiked authored analogs and spiked
reference drugs carry `chembl_id = None` and are correctly dropped, since they are not
members of the equalized pool.

## Results

| target | ref drug | register | rank on pool | n | percentile | rank in 3,417-screen |
|---|---|---|---|---|---|---|
| fabry | migalastat | **physchem** | **7** | 1933 | **0.36%** | 11 |
| fabry | migalastat | **pharmacological** | **1675** | 1933 | **86.65%** | 2974 |
| gaucher | ambroxol | physchem | **NOT_IN_POOL** | 1933 | — | 1219 |
| gaucher | ambroxol | pharmacological | **NOT_IN_POOL** | 1933 | — | 1143 |
| cf | lumacaftor | physchem | 360 | 1933 | 18.62% | 640 |
| attr | tafamidis | physchem | 177 | 1933 | 9.16% | 291 |

## Step 2 — the register finding survives the restriction

This was the thing that had to carry forward, and it does, with the gap slightly wider in
percentile terms than in the original screen:

- **physicochemical phrasing: rank 7 / 1,933 (top 0.36%)**
- **pharmacological phrasing: rank 1,675 / 1,933 (top 86.65%)**

Same disease, same weights, same compounds, same prompt template — only the register of
the condition differs, and migalastat moves by **239x in rank**. The pharmacological arm
remains worse than random on the equalized pool, as it was on the full screen.

## gaucher cannot be placed on this ruler

Ambroxol is absent from `pool_v1` entirely — not under any `chembl_id`, and not by name.
This is consistent with its known status in this project: it is not `max_phase=4` in
ChEMBL (approved in Europe, never US FDA), which is why try10 had to spike it into the
screening library with `chembl_id = None` in the first place.

It is reported as `NOT_IN_POOL` rather than patched in. Injecting it would break the
frozen, hash-verified pool contract, and the two gaucher rows are not comparable to the
other arms on this ruler under any treatment.

**Consequence for whoever assembles the cross-architecture table: gaucher drops out of
the equalized comparison for this arm.** If gaucher must be retained, the pool itself has
to be rebuilt to include non-FDA-approved compounds — a `pool_v2` decision, not something
to resolve inside this try.

## Notes on what is and is not in the CSV

- The `register` column is restricted to the two clean registers. fabry's two arms are a
  controlled pair (same condition content, deliberately reworded). gaucher's
  `pharmacological` row is its original authored condition, which is genuinely in that
  register ("modulatory contact rather than a clean mimic... modest potency") and was the
  template used to construct fabry's pharmacological arm.
- cf's and attr's **authored** conditions are *not* included as a second arm, because
  neither is cleanly pharmacological — cf's is structural but non-selective, attr's is
  partially structural. Labelling either as `pharmacological` would misrepresent them.
  For reference only, their authored full-screen ranks were cf 993 and attr 1774, against
  the physchem rewrites' 640 and 291 recorded above.
- `rank_3417` is the rank in that arm's own full screen (pool sizes were 3,420, or 3,421
  for gaucher with its spiked reference), kept for traceability. It is **not** comparable
  to `rank_pool`; the two are different rulers and the restriction changes both the
  denominator and which competitors are present.
