# try25 — re-rank existing predictions on the frozen equalized pool

## What
No training, no GPU. The saved score dicts from try23 (augmented + symptom lookalikes) and
try24 (6 new diseases) are re-ranked after dropping every drug outside the frozen pool
(`06_equalized_pool/try01`). Ranking and seed-ensembling are try24's `collate.py` logic
verbatim (rank of the mean raw score over seeds 1-3) — **only the candidate field changes**.

Pool verified before anything else: 1,933 drugs, sha256
`9d89c3ce758b7ec7c8bea2866f6448a9ea71b372f0f3c3f92c8c6b690653e308`. Job 24441350 (CPU, 4 min).

`best_cos` for the try23 runs is recomputed here from their cached training splits with the
same 4-type profile try24's audit used, so both tables use one definition rather than two.

## Result: every rank improves, and almost every one then fails the specificity control

| source | disease | drug | full field | **pool** | same drug's best rank under another model | verdict |
|---|---|---|---|---|---|---|
| try24 | **exercise-induced MH** | dantrolene | 14 | **11** | 458 (Gaucher) | **counts** |
| try24 | MH of anesthesia | dantrolene | 36 | 29 | 11 (the *other* MH node) | UNRELIABLE (cos 0.134) |
| try23 | Gaucher | miglustat | 24 | **10** | 27 (Fabry) | **counts** (narrow) |
| try23 | Fabry | migalastat | 486 | 118 | **17 (CF)** | FAILS specificity |
| try23 | Gaucher | eliglustat | 896 | 204 | **5 (Fabry)** | FAILS specificity |
| try24 | tyrosinemia | nitisinone | 3,224 | 566 | **34 (Gaucher)** | FAILS specificity |
| try23 | TTR hereditary | tafamidis | 592 | 280 | **8 (Fabry)** | FAILS specificity |
| try23 | TTR ATTRV122I / wild type | tafamidis | 340 / 138 | 155 / 91 | **8 (Fabry)** | UNRELIABLE (cos 0.147 / 0.171) |
| try23 | CF | 4 modulators | 742–918 | 264–324 | **63–96 (Fabry)** | UNRELIABLE (cos 0.159) |
| try24 | CEL | imatinib | 6,216–6,322 | 804–839 | **121 (CF)** | UNRELIABLE (cos 0.133) |
| try24 | systemic mastocytosis | midostaurin | 6,427 | 976 | 768 (Fabry) | UNRELIABLE (cos 0.177) |
| try24 | PKU | sapropterin | 599 | 211 | 778 (Fabry) | UNRELIABLE (cos 0.178) |
| try24 | NSIAD | tolvaptan | 6,373 | **not in pool_v1** | — | not evaluable |

Full per-seed numbers in `results_pool_v1.csv` (both fields side by side, `pool` labelled on
every row), ensembled + verdicts in `results_pool_v1_ensemble.csv`, and the complete
model × drug matrix in `specificity_pool_v1.csv`.

## Reading it
- **Ranks moved a lot, all in the same direction.** 7,957 → 1,933 removes mostly biologics,
  which is what had been filling the top of these lists (try12: many were isolated
  zero-score nodes anyway). Tyrosinemia 3,224 → 566 and CEL ~6,300 → ~820 are the largest
  moves. This is a change of denominator, not of evidence: the percentile barely moves
  (tyrosinemia 0.405 → 0.293, CEL 0.79 → 0.43).
- **The equalized field makes the category-level problem unmistakable.** With biologics gone,
  what remains at the top of the Fabry/Gaucher/CF/TTR models is each other's drugs.
  Tafamidis ranks **8** under the Fabry model and 155 under TTR's own. Migalastat ranks 17
  under CF and 118 under Fabry's own. This is try17's finding, now visible on a common
  field: these four models have learned *rare metabolic disease drugs*, not any one disease.
- **Malignant hyperthermia is still the only clean pass.** Dantrolene: 11 at its own node,
  458–1,223 under every unrelated model. The 29-vs-11 comparison flagged for the anesthesia
  node is against the *other MH node* — the same disease, so it is not a specificity
  failure; that node is excluded on the cos rule in any case.
- **Gaucher's miglustat "counts" only narrowly** (10 own vs 27 under Fabry). Treat it as
  borderline, not as a second clean success, and remember its lookalike is Niemann-Pick,
  which carries miglustat (try18) — the rank is still partly borrowed.

## Caveat on the control
`specificity_pass` compares a drug's own-disease rank against its best rank under every
*other* (disease, node) model, including sibling nodes of the same disease (the two MH nodes,
the three TTR nodes). A sibling is not an independent disease, so a loss to a sibling is not
evidence of non-specificity; read those rows with the sibling column in hand. Losses to
*unrelated* diseases (Fabry vs TTR, Gaucher vs tyrosinemia) are the real failures.

## Files
- `rerank_pool.py` — the whole job: pool hash check, re-rank, specificity control, verdicts.
- `rerank_pool.sb`, `rerank_pool.out`.
- `results_pool_v1.csv`, `results_pool_v1_ensemble.csv`, `specificity_pool_v1.csv`.
