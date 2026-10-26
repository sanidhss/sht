# try16 — does a better-connected disease node mean a better rank for its drugs?

## Question
try13/14 left two candidate explanations for Fabry/Gaucher working while CF/TTR are
weak: (1) **connectivity**, meaning the disease node is richly linked in PrimeKG
(Fabry 109, Gaucher 133 non-drug edges vs CF 45, ATTRV122I 3), and (2) **treated
siblings**, meaning a true drug is already indicated for a graph-close disease
(Gaucher ↔ Niemann-Pick). Four diseases can't separate these. This test uses 120,
**only for this test** (they're not new thesis targets).

## Design
- `sample_diseases.py` / `.out` / `sample.csv`: from the 1,354 diseases with ≥1
  indication, drop the thesis targets and their held-out siblings/parents, the 5
  with degree 0 (unscoreable, same as try12) and the 10 with no drug that keeps an
  edge. Then stratify into 6 log-degree bins (1–3, 4–8, 9–17, 18–37, 38–76,
  77–1168) and sample 20 per bin (sampling seed 0). That's 120 diseases with
  degrees 1–518 and 933 held-out indication edges. Fabry/Gaucher/CF/TTR all fall
  inside that range.
- `txgnn_connectivity.py` / `.sb`: same model config as every run since try11. All
  120 diseases are held out **together** (one model per seed scores all of them,
  instead of 120 separate ~8h trainings), and there's one `eval_disease_centric` call
  over all 120 per seed, so its slow preamble runs once.
  - The split-folder name is `'disease_eval_' + str(idx_list)` (`TxData.py:72`), and
    for 120 floats that's longer than the 255-char filename limit. So the holdout is
    passed as a `list` subclass (`SplitTag`) whose `str()` is
    `[connectivity_sweep_120]`. `disease_eval_fold` only needs `isinstance(list)` +
    `np.array()`, and the value is never pickled into the model config
    (`TxGNN.py:91`). No library patch.
  - Holding 120 out together removes their drug edges from each other's graphs too.
    That's stricter than holding out one at a time, and it matches how TxGNN's own
    zero-shot evaluation is run.
- `analyze.py` (CPU, run after the job). Per disease, seed-ensembled:
  - **`median_pct`** (primary): the median percentile rank of the true indications
    among all 7,957 drugs. It's ~0.50 under chance however many true drugs a disease
    has, so label count doesn't confound it.
  - **`best_cal`** (secondary): the best true drug's percentile p calibrated as
    1−(1−p)^k for k true drugs, which is uniform under chance.
  - Exact-0.0 isolated drugs are excluded (try12).
  - Covariates: degree, and **sibling path** (any true drug indicated, in TRAIN, for
    a direct disease_disease neighbour or a disease sharing ≥1 neighbour).
  - Reports Spearman(degree, outcome), per-bin medians, the sibling split, the
    degree↔sibling confound, and a rank regression `median_pct ~ degree + sibling +
    n_pos`. It overlays the thesis targets on the same scale from their own runs.
  - Mapping and outcome code were validated on the existing runs: they reproduce
    Fabry 16, Gaucher 17, CF 1,208, TTR 974.
  - Side finding from that validation: CF's best drug (1,208) is about chance level
    for the best of 9 (`best_cal` 0.77), but its median drug is at the 21st
    percentile (chance is 50th). CF's signal is broad-but-shallow, not one standout.

## Status
- Dry run (26-10-2026): split built under the short folder name, `check_blind` OK on
  all 120, all 120 present in test (933 indication, 1,353 contraindication, 178
  off-label rows).
- **Submitted 26-10-2026 as job 23210549** (24h limit; it scores 120 diseases instead
  of 1, so it may run longer than the ~8h of earlier runs). Results pending. Next: run
  `analyze.py`, then plot degree vs `median_pct` with the sibling flag and the targets
  overlaid.
