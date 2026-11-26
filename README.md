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
  pass/fail but a **triage rule** — fine-tuning drags every target onto a fixed
  ~30th-percentile band wherever it started (`r(zero-shot, improvement) = +0.867`,
  `r(zero-shot, final) = -0.216`), so the deciding variable is the base model's
  zero-shot percentile, measurable in one GPU-hour before committing to training. A
  strong prior is usually *destroyed* by fine-tuning (2 of 3 targets); a cold start is
  reliably helped, ordered by data volume. **try16 completed 25-11 on the frozen
  equalized pool**, all six evaluable targets: **2 genuine wins** (cel 10.28 vs kNN
  17.80, cf 38.89 vs 71.03), 1 rank artifact (fabry — it outranks the kNN only because
  the kNN is more wrong in the same direction), 3 clear losses, the worst being
  mastocytosis at 63.34 against a kNN at **0.57**. That is roughly even, against 1 win /
  1 tie / 6 losses on the old 500-decoy field.
  `NEXT_PHASE_diverse_structural_training.md` is **superseded** — try13 showed the
  fine-tuning data is not where the discrimination comes from.
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
  pool, out of thirteen target nodes: **three pass the specificity control** (MH,
  Gaucher, PKU) but only two also pass `reliable`, so the headline is **one clean pass**
  — dantrolene at 11/1,933 for malignant hyperthermia (458+ under every unrelated
  model) — plus Gaucher/miglustat at 10 vs 27, borderline, and PKU/sapropterin specific
  (211 vs 778) but unreliable. **The model ranks
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

**Both blinded arms are now on one ruler.** `06_equalized_pool`'s 1,933-drug pool is the
shared field: the graph arm re-ranked onto it in 03/try25 (23-11) and the transformer arm
completed all six evaluable targets in 01/try16 (25-11). The side-by-side per-disease
percentiles are **held locally and deliberately not committed to this repository**; each
arm's own numbers stay in its track README. Read them with the controls attached, not as
a scoreboard: TxGNN leads on most shared diseases **but fails the specificity control on
all of them except malignant hyperthermia** (its drugs rank as well or better under
unrelated disease models), and TxGemma's weaker rows have seed spreads wide enough to
swallow their own means. Chance is 49.9%.

**Still missing:** unblinded ceilings for both arms (01/try17, 03/try26) are queued, and
**Gaucher cannot be placed on the shared ruler at all** — ambroxol is absent from the
pool — which bears directly on the thesis's stated Fabry+Gaucher proof-of-concept pair
and needs an explicit decision: drop it, or re-freeze as `pool_v2` with a curated
override table, reported as such.

**Standing decisions, unchanged.** Do *not* merge chemistry into TxGNN's architecture —
it would blur the clean architectural contrast and is a large, risky library
modification; ensembling the two arms' outputs is the right move if either arm earns it.
Do *not* keep deepening Fabry-specific fixes in isolation (stereochemistry- or
diastereomer-specific training): it risks teaching a narrow heuristic for lysosomal
enzymes with no evidence it extrapolates, which is the actual thing this thesis needs.
try13 strengthened the second decision — the diastereomer collapse is a **base-model
property**, present with zero training compounds involved, so there was never a
data-side fix for it.
