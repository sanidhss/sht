# try19 — option 1: per-drug popularity correction (no retraining)

## Idea
try17 showed rankings are category-level: related diseases get near-identical lists,
and generic drugs (steroids, avelumab, ...) top unrelated diseases. This test
re-scores each drug as
`z(drug, D) = (score(drug, D) − mean_bg(drug)) / std_bg(drug)`, using the **same
model's** scores for that drug across many other diseases. A drug then ranks high only
if it is unusually high for D specifically. The 2026 bioRxiv preprint on
topology/degree bias in biomedical KGs describes the same failure (high-degree drugs
favoured).

## Files
- `make_background.py` → `background.csv`: 150 background diseases (≥1 indication,
  ≥1 non-drug edge; thesis targets and siblings/parents excluded; seed 1).
- `background_infer.py` / `.sb`: **inference only**. Reloads each thesis run's saved
  checkpoints (fabry, gaucher, cf, ttr, gaucher_no_np × 3 seeds) and scores the 150
  background diseases. It skips seeds that are already done or not yet trained, so
  it's safe to resubmit after try15 finishes.
- `rerank.py`: CPU. Thesis mode uses the 150-disease background. Sweep mode uses a
  leave-one-out background over try16's 120 diseases (and try21's, when done).
  Reports raw → corrected ranks and the new top 10.

## Important: reloaded models must reproduce the original scores
Node **input** vectors are random, frozen, and stored on the graph
(`utils.py` L1076 `initialize_node_embedding`, `requires_grad=False`). They are
**not saved in the checkpoint**. A reload matches the trained model only if torch's
RNG is in the same state when they're drawn. The training script re-seeds right
before `TxGNN()`, so `background_infer.py` re-seeds at the same point. It then
**re-scores the run's own eval disease and aborts if any drug's score differs from
the saved one by more than 1e-3**, so a silent mismatch can't produce a bogus
background.

(Correction to try12/CLAUDE.md wording: node inputs are random frozen vectors, not
"learned embeddings". The learning is in the layer weights. try12's conclusion,
isolated nodes → zero vector → exact 0.0, is unaffected.)

## Status
- Job **23231183** submitted 26-10-2026 (12h). `rerank.py` was smoke-tested and
  reports "not available yet" for every input until the jobs finish.
