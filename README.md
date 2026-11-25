# Overview

A senior honors thesis project testing whether current AI models can nominate small
molecules that stabilise misfolded proteins in orphan diseases — and, more importantly,
whether their predictions hold up when checked against structure and energetics.

## Why three models

Three architectures are run independently, chosen because they draw on **disjoint**
evidence:

| arm | sees | never sees |
|---|---|---|
| knowledge-graph link prediction | disease/gene/drug relationships | any chemical structure |
| numeric drug-target scoring | molecular structure + measured bioactivity | the graph |
| reasoning / free-text screening | natural-language mechanism descriptions | either of the above |

Because the inputs don't overlap, agreement between arms is meaningful and disagreement
is diagnostic. Each arm is held blind to the validated reference compound for the target
being evaluated.

# Library index

Organized by project, then by try (one folder per distinct attempt, named
`tryNN_DD-MM-YYYY_short-description`). Each try folder contains the scripts as they
existed at that point (where separately preserved — several were edited in place across
iterations and only the final version survives; noted explicitly where that's the case),
the real SLURM job log(s), and a `README.md` explaining what it does and, if it failed,
why.

Model weights, PrimeKG's database, HuggingFace cache, and other large data artifacts are
**not** duplicated here — they remain at their original paths under
`/work/hdd/bisa/sshah21/` (`hf_cache/`, `primekg/`, `results/`). This library holds code
and logs only.

## Projects

- **[01_txgemma_predict](01_txgemma_predict/README.md)** — numeric drug-target scoring
  fine-tune (`txgemma-9b-predict`), chemistry only (SMILES + real bioactivity data).
  17 tries, 8 targets. Current state: the defensible result is **not** a per-disease
  pass/fail but a **screening criterion** — whether the pipeline will work on a given
  orphan drug is measurable zero-shot in one GPU-hour (try13), and it sorted all four
  original targets correctly. Fine-tuning produces swings up to 80 percentile points in
  *both* directions and converges every target to one band, i.e. it teaches the task's
  output distribution more than the target's chemistry (try14/15). Now moving onto the
  frozen equalized pool (try16/17): **2 of 6 targets landed**, cf and fabry, both
  beating their Tanimoto baseline but with cf's seed spread at 43.8 points and fabry's
  baseline inadmissible (leak-flagged). `NEXT_PHASE_diverse_structural_training.md` is
  **superseded** — try13 showed the fine-tuning data is not where the discrimination
  comes from.
- **[02_txgemma_chat](02_txgemma_chat/README.md)** — reasoning-based chaperone
  identification (`txgemma-27b-chat`), leave-one-target-out across 4 mechanism
  classes. 12 tries. **Descoped from the architecture comparison** (try12); it is kept
  for one causal finding: holding disease, weights and library fixed and changing only
  the *register* of the stated necessary-condition moves migalastat from rank 7/1,933
  to 1,675/1,933 — a 239x swing from phrasing alone. End-to-end identification for an
  unseen disease is **not supported** (try09 misses by ~330x), a deterministic RDKit
  property filter beats the model on all four folds by 2-48x (try11), and no self-check
  has survived validation. Open and unsolved: stage 1, deriving the condition without
  hallucinating it.
- **[03_txgnn](03_txgnn/README.md)** — PrimeKG knowledge-graph link prediction, no
  chemistry at all. 26 tries, the longest debugging arc. Current state on the equalized
  pool with the specificity control applied: **one clean pass out of thirteen target
  nodes** — dantrolene at 11/1,933 for malignant hyperthermia (458+ under every
  unrelated model) — plus Gaucher/miglustat at 10 vs 27, borderline. **The model ranks
  by disease *category*, not by disease**: tafamidis ranks 8 under the *Fabry* model and
  155 under TTR's own. Fabry's old top-0.2% result is **void** — try18 found migalastat's
  answer in the training graph under another name (a *chylomicron retention disease*
  duplicate carrying all 23 Fabry genes and a migalastat indication).
- **[04_setup](04_setup/README.md)** — one-time environment/data setup (model weights,
  PrimeKG). 2 tries, both resolved cleanly.
- **[05_multi_agent](05_multi_agent/README.md)** — context-budget audit and
  prefilter recall@k for a multi-agent screening protocol. 2 tries. Supplies the
  Tanimoto 5-NN estimator that now serves as the comparison baseline in 01.
- **[06_equalized_pool](06_equalized_pool/try01_23-11-2026_build-pool/README.md)** —
  **the common ruler, and the prerequisite for any cross-architecture claim.** Builds
  and freezes `pool_v1.json`: the 1:1 ChEMBL↔DrugBank intersection of TxGNN's 7,957
  PrimeKG drug nodes and the chat track's 3,417-drug approved library = **1,933 drugs**,
  sha256 `9d89c3ce…`, byte-stable, hash-verified inside every arm's scoring script.
  1 try. **Two targets are unevaluable on it** and nothing was substituted: ambroxol
  (Gaucher) is not `max_phase=4` in ChEMBL, and tolvaptan's UniChem partner is a
  different ChEMBL record from the library's.
- **[07_structure_pipeline](07_structure_pipeline/README.md)** — the structural /
  biophysical half of the thesis (Boltz-2 and AlphaFold3 complex prediction feeding
  interface energetics). **Setup and smoke only; `boltz2_smoke` and `af3_smoke` are
  still PENDING in the queue.** No structural result exists yet.

## Cross-project context
All three model tracks (01/02/03) are independent attempts at the same underlying
question: can a computational tool correctly identify pharmacological
chaperones/correctors/stabilizers for orphan disease targets (Fabry, Gaucher, and later
cystic fibrosis and TTR amyloidosis) with genuine, verifiable blindness to the true
answer during training/evaluation — not just easy discrimination against obviously
unrelated compounds. The "hard negative" discipline (real structurally-similar
analogs, not just unrelated drugs) recurs across all three as the thing that separates
a real result from an easy, uninformative one.

### RETRACTED: the "complementarity" finding this README used to lead with

> This README previously led with: *TxGemma-predict passes on Gaucher and fails on
> Fabry; TxGNN does the opposite — strong on Fabry, weaker on Gaucher; the two signal
> sources are therefore complementary.* **Both halves of that contrast have since been
> withdrawn by the tracks that produced them, so the inference is void.**
>
> - **TxGemma's Gaucher pass:** every pre-try09 verdict in 01 was produced by a leaky
>   test *and* an over-trained model — all 12 hard negatives were in the training sets,
>   and `best_epoch = 0` showed the pipeline trained ~3x past optimum with no validation
>   set at all. try13 then showed what remains of Gaucher's result is a **base-model
>   prior that LoRA sharpened**, not knowledge learned from its 1,208 training
>   compounds.
> - **TxGNN's Fabry result:** try18 found migalastat's answer sitting in the training
>   graph under another name (a *chylomicron retention disease* node carrying all 23 of
>   Fabry's genes and a migalastat indication), and try17 showed the model ranks by
>   disease category anyway — migalastat scores 16 under the Fabry model and 27 under
>   Gaucher's.
>
> Kept visible rather than deleted, because the pattern of **a clean story arriving
> before the leak audit that kills it** is the recurring shape of this project and the
> reason the audits are now run first.

**What actually spans the tracks is methodological, and it is the project's most
transferable output:** every apparent success here was produced by a measurement
artifact until the measurement was rebuilt, and each one was caught by a deliberate
control rather than by inspection.

| artifact | caught by | what it had produced |
|---|---|---|
| hard negatives present in training | 01/try08 | every pre-try09 pass/fail |
| ~3x over-training, no validation set | 01/try09 | the memorization signature |
| greedy decoding returning modes, not estimates | 01/try10 | every enrichment number to that point |
| percentiles depending on the decoy pool (31-point swings) | 01/try15 | all cross-try comparisons |
| the answer in the graph under a different disease name | 03/try18 | Fabry's top-0.2% |
| category-level ranking mistaken for disease-level | 03/try17, 03/try25 | Fabry *and* Gaucher |
| eligibility inferred from a holdout list, not the actual split | 03/try26 | would have mis-scored 14% of cells |
| hard-negative *selection* picking salts/tautomers of the reference | 01/try14 | 3 of 4 new targets' verdicts |

**The cross-architecture comparison is now on one ruler but is not finished.**
`06_equalized_pool`'s 1,933-drug pool is the shared field; the graph arm is complete on
it (03/try25) and the transformer arm has 2 of 6 targets (01/try16). Unblinded ceilings
for both arms (01/try17, 03/try26) are queued. **Gaucher cannot be placed on the shared
ruler at all** — ambroxol is absent from the pool — which bears directly on the thesis's
stated Fabry+Gaucher proof-of-concept pair and needs an explicit decision: drop it, or
re-freeze as `pool_v2` with a curated override table, reported as such.

**Standing decisions, unchanged.** Do *not* merge chemistry into TxGNN's architecture —
it would blur the clean architectural contrast and is a large, risky library
modification; ensembling the two arms' outputs is the right move if either arm earns it.
Do *not* keep deepening Fabry-specific fixes in isolation (stereochemistry- or
diastereomer-specific training): it risks teaching a narrow heuristic for lysosomal
enzymes with no evidence it extrapolates, which is the actual thing this thesis needs.
try13 strengthened the second decision — the diastereomer collapse is a **base-model
property**, present with zero training compounds involved, so there was never a
data-side fix for it.
