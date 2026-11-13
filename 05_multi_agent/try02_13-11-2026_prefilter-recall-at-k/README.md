# Try 02 — prefilter recall at k

**Date:** 13-11-2026
**SLURM job:** prefilter_recall (23850681), CPU, COMPLETED
**Scripts (exact versions run, snapshotted at submission time):**
- `prefilter_recall.py`
- `slurm_script`

## What it does

Sets `k` for the multi-agent protocol by measuring it, because the paper never reports
its own top-k and its setting cannot be inherited: RareAgent's candidate space is **74
drugs**, this project's library is **3,420** — about 46x larger. In RareAgent, "debate
the top k" is close to debating everything. Here, something must cut 3,420 down to k
first, and **if the reference drug is not in that top-k, no amount of debate recovers
it.** The pre-filter's recall@k is therefore a hard upper bound on the entire arm.

Pre-filter measured: Tanimoto 5-NN over Morgan-2 2048-bit fingerprints, predicting
potency as the similarity-weighted mean label of the 5 nearest compounds with measured
activity against that target. Same family as `01_txgemma_predict`'s
`descriptor_baseline_predict.py` — the baseline that already beats the fine-tuned 9B on
5 of 6 targets. Actives come from `train_blind.jsonl`, which already holds the reference
out; the script asserts that rather than assuming it.

## What happened

**The chemistry pre-filter does not work as a top-k selector. recall@20 is 1/8.**

| target | reference | rank / 3,417 | pct | actives | max Tanimoto to actives | leak |
|---|---|---|---|---|---|---|
| mastocytosis | midostaurin | **17** | 0.50% | 1,501 | 0.692 | |
| nsiad | tolvaptan | 344 | 10.07% | 1,109 | **1.000** | TRIVIAL |
| cel | imatinib | 581 | 17.00% | 1,501 | 0.942 | close |
| tyrosinemia | nitisinone | 621 | 18.17% | 179 | 0.633 | |
| gaucher | ambroxol | 1,879 | 54.97% | 1,208 | 0.192 | |
| attr | tafamidis | 1,921 | 56.22% | 493 | 0.435 | |
| cf | lumacaftor | 2,455 | 71.85% | 1,450 | 0.575 | |
| fabry | migalastat | 2,964 | 86.74% | 89 | **1.000** | TRIVIAL |

Recall@k across the 8 targets:

| k | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|
| recall | 0% | 12% | 12% | 12% | 12% | 25% | 50% | 75% |

Four of the 12 pins are absent entirely — PKU (2 pchembl records), propionic acidemia
(4), MH (10), HLCS (11) have too few measured activities to support any chemistry
pre-filter. That absence is the result for those cells, not a gap in the run.

### Why this is the most consequential result so far

A seed-top-k-then-debate pipeline built on this filter would **miss the answer before
the debate starts in 7 of 8 cases**. Three of the targets land in the bottom half of the
library. No choice of k between 10 and 200 changes anything: recall is flat at 12%
across that whole range, because only mastocytosis is anywhere near the top.

So the question "what k?" has the answer "k is not the problem." try01 worried about
whether k=50 fits in context; that worry is now moot, because k=50 catches 1 target in
8.

### The two TRIVIAL flags, and why Fabry is the interesting one

The leak control asks: how similar is the nearest *active* to the held-out reference?
Both Fabry and NSIAD return **1.000** — a fingerprint-identical compound is in the
actives pool despite the reference itself being held out.

Fabry is the diagnostic case. It has 3 actives at similarity ≥0.95, an identical
fingerprint to migalastat, **and still ranks 2,964/3,417.** The reason is that default
Morgan fingerprints ignore stereochemistry, so migalastat and its diastereomers are
*literally the same fingerprint* while their real labels are 196 / 925 / 754 / 993. The
k-NN averages four wildly different potencies attached to one identical structure and
predicts something middling, which ranks badly.

This is the same stereochemistry collapse `01_txgemma_predict` diagnosed in the 9B model
(try11/try13: "collapses to predicting the same score for migalastat and all 3 of its
real diastereomers"). **It is not a property of the LLM.** It is a property of
structure-only representations of this target, and it reproduces in a model with no
neural network in it at all. That is a cleaner statement of the Fabry failure than the
project currently has, and it is worth carrying back to 01.

### A number that should not be compared carelessly

This filter ranks against **3,417 real approved drugs**. `01`'s enrichment percentile
ranks against **500 MaxMin-picked decoys drawn from other targets' training sets**.
Those are different difficulties — an approved-drug library contains many compounds
genuinely similar to the actives, while cross-target decoys are drug-like but
unrelated. A filter that looks strong against decoys can be weak here, which is exactly
what happened. **Do not quote these ranks against `01`'s percentiles.**

Likewise, `02_txgemma_chat` try11's "RDKit beats the model 2–48x" is a *different* RDKit
baseline — a hand-authored property filter scored as rank-within-the-kept-set. It is not
this filter and these numbers neither confirm nor contradict it.

## What this changes

**The seeding stage must not be chemistry k-NN.** Options, in the order they should be
considered:

1. **Cheap LLM screen as the pre-filter** — one short forward pass per compound over all
   3,420 (exactly `02_txgemma_chat`'s yes/no log-prob margin design), then debate its
   top-k. This is cheap under vLLM (~41k short calls for 12 cases) and it makes the
   comparison clean: the single-shot screen *is* the control cell of the 2×2, and the
   multi-agent arm is that same screen plus debate on its top-k. Note that bare TxGemma's
   margin screen put migalastat at **rank 4** where this chemistry filter puts it at
   2,964 — the signals are very different, whatever the memorisation caveat.
2. **Union seeding** — top-k from the chemistry filter *and* top-k from the LLM screen,
   deduplicated. Keeps a chemistry path into the candidate set without letting it gate.
3. Chemistry k-NN stays valuable as a **per-candidate tool inside the debate** (it is
   real measured evidence about one compound) — just not as the thing that chooses who
   gets debated.

**Before building stage 2, measure stage 1's recall@k the same way this try measured the
chemistry filter's.** If the LLM screen's recall@k is also flat and low, the whole
top-k-then-debate architecture is wrong for a 3,420-compound library and the design has
to change rather than be tuned.

## Caveats

- One pre-filter, one parameterisation (`KNN_K=5`, Morgan-2 2048-bit, no chirality). A
  chirality-aware fingerprint would likely change Fabry specifically and is worth one
  run, though it cannot fix the 7-of-8 pattern.
- `n_actives` differs ~17x across targets (89 for Fabry vs 1,501 for cel/mastocytosis),
  so these ranks are not a controlled comparison between targets.
- Mastocytosis's rank of 17 is the one success and should be treated cautiously: try14
  recorded that its hard negatives are *more potent* than midostaurin, meaning its
  chemistry neighbourhood is unusually rich and well measured (1,501 actives).

---
## Auto-recorded outcome (finalize_trial.sh)
**SLURM state / elapsed / exit code:** RUNNING 00:01:20 0:0
**Log file:** prefilter_recall_23850681.out

_If this failed, the root-cause diagnosis above this line still needs to be_
_written by hand -- this section only records the raw outcome, not why. See_
_library/AGENT_INSTRUCTIONS.txt for who's expected to fill that in and when._
