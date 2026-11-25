# try26 — unblinded ranks for every disease, on the frozen equalized pool

## What this asks
Every number so far is **blinded**: the disease's `indication` edges are held out, so the
model has to place the right drug without ever having seen it attached to that disease.
When a blinded rank is bad, two very different things could be true:

1. the graph does not contain the signal (no path from this disease to this drug), or
2. the model cannot express it even when handed the answer.

The unblinded rank separates those. If the model ranks the drug at ~1 when the edge **is**
in its training graph, (2) is ruled out and the blinded failure is a real transfer failure.
If it is still bad unblinded, the architecture cannot even memorise the edge and no blinded
result from it means anything. This is a **ceiling**, not a prediction.

## No retraining was needed
Each try23/try24 run held out only its own disease. Every *other* disease's indication edges
are therefore already in that run's TRAIN split. So scoring disease D with a model that did
not hold out D is an unblinded evaluation, obtainable by inference on the existing
checkpoints — the same trick try19 used for its 150 background diseases. Nothing in try23
or try24 is modified or retrained; `txgnn` is not reinstalled.

Because several models are usually eligible for one target, each target gets a *spread* of
unblinded ranks rather than one number, which is more informative than a single run.

## Jobs
`A_edge_audit.sb` (CPU) → `B_infer.sb` (GPU array 0-9, `afterok`) → `C_collate.sb`
(CPU, `afterany`). Layout frozen before submitting (try06).

- **A — `edge_audit.py`.** For every (model, target node, reference drug), reads that
  model's cached split and records which file holds the edge: `train` / `valid` / `test` /
  `absent`. This is the basis of the whole try, so it is verified edge by edge rather than
  inferred from the holdout list. Only `train` counts as unblinded: valid and test edges are
  kept out of the message-passing graph, so a target sitting in `valid` under some model is
  still blind there. Also writes `node_ids.csv`, the x_idx → raw-PrimeKG-id map that
  `eval_disease_centric()`'s prediction dict is keyed by (try10's third id system).
- **B — `unblind_infer.py`.** One task per model, 3 seeds each. Reloads the checkpoint and
  scores all target nodes in one pass. Node *input* vectors are random, frozen, stored on
  the graph and **not** in the checkpoint, so each seed is re-seeded immediately before
  `TxGNN()` exactly as the training script did, and the reload is then **verified** by
  re-scoring that run's own held-out disease against its saved `eval_*.pkl`
  (`max |score diff| < 1e-3`, try19's guard). A mismatch aborts the task rather than
  reporting numbers from a silently different model.
- **C — `collate26.py`.** Verifies `pool_v1.sha256` or stops, restricts to the 1,933-drug
  pool, ranks with try24's `rank_of()` and the same seed ensembling (rank of the mean raw
  score over seeds 1-3), and joins try25's blinded ranks for the same (node, drug). Both
  fields (7,957 and 1,933) are kept side by side.

## Reporting rule — fixed before any unblinded number was looked at
- A cell counts as unblinded only if job A finds that exact edge in that split's `train`.
- Headline unblinded rank = **median across all eligible models** of the seed-ensembled
  rank; `best_model_rank` / `worst_model_rank` / `n_models` are printed beside it. Taking
  the best-looking model would be selecting on the outcome (try24's rationale).
- try20's UNRELIABLE rule is **not** applied to the unblinded numbers. That rule flags a
  weak prototype blend, and the prototype path barely applies here: `agg_measure='rarity'`
  weights the blend by the node's degree in the training graph, so an unblinded disease
  (which keeps all its edges) is scored mostly from its own learned embedding. The blinded
  `best_cos` and `reliable` are carried into the table for reference only.

## Known before running — two TTR nodes have no answer edge at all
Checked directly on `primekg_aug/kg_directed.csv`: tafamidis's only `indication` edges go to
**6669 ATTRV122I**, 6361 *amyloidosis (disease)*, 6665 *AL amyloidosis* and 6008 *AH
amyloidosis*. There is **no tafamidis edge at 5528 (wild type ATTR) or 4803 (hereditary
ATTR)**. So:
- Those two nodes can never be evaluated unblinded — there is no edge to put in train. Job B
  skips them; job A still records the absence.
- try23/try25's ranks at those two nodes (pool 91 and 280) were ranks **with no ground-truth
  edge behind them**. They should not be read as hits or misses. ATTRV122I is TTR's only
  properly labelled node, and 6665/6008 are PrimeKG data errors (tafamidis is not indicated
  for AL or AH amyloidosis), so they are not added as targets.

## Result — job A in, jobs B/C pending (25-11-2026)

**Job A (`24533464`, edge audit) COMPLETED 25-11 13:06**, 1m13s.
`/work/hdd/bisa/sshah21/results/txgnn_v5_unblinded/edge_audit.csv`, 180 rows =
10 models x 18 (target node, reference drug) cells, plus `node_ids.csv` (the
x_idx → raw-PrimeKG-id map `eval_disease_centric()`'s prediction dict is keyed by).

| `edge_in` | cells | counts as unblinded? |
|---|---|---|
| `train` | **137** | yes |
| `absent` | 20 | no — no edge exists |
| `test` | 16 | no — held out, not in message passing |
| `valid` | 7 | no — held out, not in message passing |

**The valid/test split mattered, and would have been missed by inferring eligibility
from the holdout list.** 23 cells sit in `valid` or `test` under some model: a target can
be absent from a run's holdout list and still be blind there, because the random
valid/test draw took its edge out of the message-passing graph anyway. Those 23 are
correctly excluded. Had eligibility been computed as "not in this run's holdout list" —
the obvious shortcut — 23 of 160 edge-bearing cells (14%) would have been scored as
unblinded while the model never saw the edge.

### The two TTR nodes are confirmed unevaluable, 0 of 10 models

| target node | drug | models giving an unblinded cell |
|---|---|---|
| exercise-induced MH / MH of anesthesia | dantrolene | 9/10 each |
| Fabry disease | migalastat | 9/10 |
| Gaucher disease | miglustat, eliglustat | 9/10 each |
| cystic fibrosis | all 4 modulators | 9/10 each |
| ATTRV122I amyloidosis | tafamidis | 9/10 |
| NSIAD | tolvaptan | 9/10 |
| chronic eosinophilic leukemia (2 nodes) | imatinib | 9/10, 7/10 |
| phenylketonuria | sapropterin | 8/10 |
| tyrosinemia | nitisinone | 8/10 |
| systemic mastocytosis | midostaurin | 6/10 |
| **hereditary ATTR amyloidosis** | tafamidis | **0/10** |
| **wild type ATTR amyloidosis** | tafamidis | **0/10** |

This is the prediction in "Known before running" above, now verified by the audit rather
than assumed: there is no tafamidis `indication` edge at 4803 or 5528, so no split can
place one in `train` and no model can be unblinded there. **try23/try25's pool ranks of
280 (hereditary) and 91 (wild type) are ranks with no ground-truth edge behind them and
must not be read as hits or misses** — including the 91, which is the better-looking of
the two and therefore the one most likely to be quoted by mistake.

9/10 is the expected ceiling, not a shortfall: each cell is blind under exactly the one
model that held that disease out. The cells below 9 (sm 6/10, cel's second node 7/10,
pku and tyr1 8/10) are the valid/test draws above.

**Jobs B (`24533465_[0-9]`, GPU array) and C (`24533466`) are PENDING on priority** —
Delta had 2,602 jobs queued on `gpuA100x4` at submission. `24537102_[0-9]`, the A40
duplicate, was cancelled for the `PriorityTier` reason recorded in the scheduling notes
below. No unblinded rank exists yet, and the reporting rule above was fixed before job A
ran, so it cannot be tuned to the outcome.

## Scheduling notes (25-11-2026)
Delta was heavily congested when this was submitted (2,602 jobs pending on `gpuA100x4`;
4 of 392 A40 GPUs and 6 of 396 A100 GPUs actually free). Three things were changed on the
**pending** array with `scontrol update`, which preserves its queue age and priority and
leaves job C's dependency intact — cancelling and resubmitting would have cost both:

1. `-p gpuA100x4,gpuA40x4`. Each task is one forward pass on one GPU, so an A40 serves as
   well as an A100 here; try24's A40 problem was *training* hitting the 12h wall, which does
   not apply. Both partitions are `PriorityTier=100`.
2. Wall time 6h → 4h. try19's comparable inference job (15 model-seed reloads) took 7h39m,
   i.e. ~30 min per reload, nearly all of it in `prepare_split` + graph construction rather
   than scoring. One task here is 3 reloads ≈ 1.5h, so 4h keeps ~2.5x headroom while being
   far easier for the backfill scheduler to place than 6h.
3. The `gpuA40x4-preempt` duplicate was withdrawn. The `-preempt` partitions are
   `PriorityTier=30`, and Slurm treats the tier as a hard gate: nothing in a lower tier is
   scheduled while any higher-tier job is pending. Cluster-wide, zero jobs were running in
   any `-preempt` partition. `sinfo` showing those nodes as `mixed` is misleading — it
   refers to free CPU cores; `scontrol show node` reports `gres/gpu=4` of 4 allocated, and
   the `-` suffix is `PLANNED` (already earmarked for a pending higher-priority job).
