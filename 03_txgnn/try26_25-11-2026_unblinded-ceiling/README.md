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

## Result
*(filled in by job C)*
