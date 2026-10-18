# Project: TxGemma-Predict (numeric drug-target scoring)

**Question:** Can `txgemma-9b-predict`'s `BindingDB_ki`-style numeric scoring correctly
rank a known drug (migalastat/GLA, ambroxol/GBA1) above unrelated and structurally
near-identical decoys, for orphan disease targets?

## Arc
1. **try01/try02** — zero-shot: failed. Root cause is data coverage, not a broken
   pipeline: BindingDB has 0 records for GLA, 17 for GBA1, and the one real ABL1/imatinib
   hit is filed under a different sequence identity (BCR-ABL fusion) than what was asked.
2. **try03** — first LoRA fine-tune on real ChEMBL data: looked like a win (migalastat
   #1 of 7), but the test used only easy, unrelated negative drugs.
3. **try04** — added real hard negatives (structural near-misses): the try03 win didn't
   hold up. Diagnosed as SMILES-string memorization, not real structure-activity
   learning.
4. **try05 (current)** — added SMILES augmentation + more data sources (BindingDB,
   PubChem BioAssay) + EC50 support. Gaucher now genuinely passes the hard-negative
   test; Fabry improved but still fails (ties rather than misidentifies). A scope bug
   (shared target list with the chat-reasoning project) caused this run to also attempt
   2 unintended targets and time out before finishing.

5. **try06** — fixed the scope-coupling bug (`TARGETS_TO_RUN` env var, both
   scripts) and ran CF/TTR as their own deliberate, isolated run through the
   *unmodified* pipeline — the real test of whether the general approach extrapolates
   beyond lysosomal-enzyme chaperones. Job 22546141 COMPLETED: **CF passes** (beats
   all 3 real structural analogs), **TTR fails** (same one-score collapse as Fabry,
   and additionally loses to easy negatives). See try06's Result for the diagnosis.

6. **try07** — tests whether the Fabry/TTR failures are about training-data
   volume by retraining **Gaucher** (passes at 1,208) on a random **493**-compound
   subset, matching TTR's count exactly. Same target, mechanism, pipeline, reference
   and hard negatives; volume is the only variable, which breaks the confound in the
   4-target pattern. 3 seeds. Job 22660248 COMPLETED: **0/3 collapsed, 2/3 passed**
   — the volume hypothesis is dead. Exposed instead that the reference prediction swings
   785/408/480 across seeds (real 685), i.e. single-run verdicts are noise-dominated.

7. **try08** — output-resolution probe. A measurement, not a fix: loads the
   four already-saved adapters and counts how many *distinct* scores each can emit
   across ~310 molecules (200 MaxMin-diverse and novel, 50 held-in as a memorization
   control, 50 reference-neighbours, 9 eval anchors). The 9-molecule hard-negative test
   cannot distinguish "those molecules really are similar" from "the model only has a
   few output values." If the score scale is coarse, small gaps like Gaucher's 691-vs-661
   are rounding, library ranking collapses into a few tie-groups, and more data cannot
   fix it. Inference only, no training. Job 22660579 COMPLETED: **no quantization
   ceiling** (min_gap=1 everywhere), but the memorization control fired on all four
   adapters, and it surfaced that **all 12 hard negatives are in their training sets**.

## Current state

**Read this before trusting any earlier pass/fail claim in this repo.** try07 and try08
(both COMPLETED 05-06/10/2026) invalidated two framings and found a flaw in the test
itself.

### 1. The hard-negative test is not measuring what it says it measures
**All 12 hard negatives, across all four targets, are in their training sets.** Only the
reference drug is held out (verified directly against each `train_blind.jsonl`). So the
test compares a **generalization** task (predict a held-out drug) against a
**memorization** task (recall three trained analogs). The script describes this check as
"the test that actually indicates the model learned the right structure-activity signal";
it cannot be that.

This explains the "collapse" signature that drove the Fabry and TTR diagnoses. TTR
emitting 781 for the reference and all three analogs is the model applying analog #1's
memorized label (real label 781) to the whole structural family — not a stereochemistry
or mechanism failure. **Fabry's diastereomer diagnosis, carried for weeks, was most
likely an artifact of this.**

### 2. Training-data volume is not the driver
Gaucher retrained on 493 compounds — exactly TTR's count — **did not collapse and passed
2 of 3 seeds**. The monotonic n_training pattern (CF 1,427 / Gaucher 1,208 / TTR 493 /
Fabry 89) is confounded or coincidental. Do not resurrect the volume framing without new
evidence.

### 3. Single-run verdicts are noise-dominated
The same target, data size, pipeline and hyperparameters — varying only *which* 493
compounds were drawn — predicted ambroxol at **785, 408, and 480** (real label 685,
full-size run 691). A 377-point swing on a 0-1000 scale. Every existing verdict in this
project is one draw from an unmeasured distribution, and two of the three seeds above
land on opposite sides of pass/fail.

### 4. There is no quantization ceiling, but there is severe output degeneracy
`min_nonzero_gap = 1` for all four adapters, so the score scale is fine-grained and small
gaps are not rounding artifacts. However, every adapter resolves *trained* compounds far
better than novel ones (held-in vs novel: Gaucher 74%/27.5%, Fabry 62%/23%, TTR
78%/47.2%, CF 72%/49%) — recall, not learned reasoning. And the attractor values are
memorized hard-negative labels: **Gaucher emits 791 for 51% of 200 diverse novel
compounds**, and 791 is the real label of one of its own hard negatives.

### Where each target actually stands
- **CF** — the only target that looks genuinely good. Highest novel-compound resolution
  (49.0%), flattest output distribution (most common value covers just 6.5% of novel
  compounds), passes both easy and hard checks.
- **Gaucher** — recorded as passing; the pass is largely an artifact. Ambroxol (691) landed
  below the 791 value the model emits for half of chemical space. It also loses to
  metformin (691 vs 661), and `beat_easy` is computed but never used in the verdict.
- **TTR** — recorded as failing, but resolves novel chemistry nearly as well as CF (47.2%
  vs 49.0%). Its failure is the memorized-label mechanism, not weak learning.
- **Fabry** — weakest on every resolution measure (23.0% novel). Its diastereomer
  diagnosis needs re-testing against a leak-free eval before being believed.

Note resolution does **not** track pass/fail: TTR fails while out-resolving Gaucher, which
passes.

### Next steps, in order
1. **Fix the eval before any more training.** Hold hard negatives out of training, and
   report multi-seed distributions instead of single-run verdicts. Everything downstream
   depends on this, and more training runs under the current test produce more unreliable
   single draws.
2. **Decide whether `beat_easy` should gate the verdict.** Currently it is computed,
   printed, saved and ignored — which is how Gaucher printed "trust this target adapter for
   broader DrugBank-wide screening" while scoring worse than metformin. Open question: is a
   hard-negative pass alone ever sufficient?
3. **Re-run the four targets under the fixed eval** before interpreting any of them.
4. `NEXT_PHASE_diverse_structural_training.md` remains **pending re-justification**, now on
   two counts: CF's pass already undercut its mechanism premise, and the failures it was
   designed to address may be eval artifacts.

### Caveats on earlier numbers
- Gaucher's analog real labels are 791/**679**/839, not the 791/791/839 quoted in places
  (those are *predicted*). Analog #2's real 679 is more potent than ambroxol's real 685, so
  "reference beats all hard negatives" is not strictly ground truth for Gaucher.
- CF's training file yields **1,450** distinct canonical compounds vs `coverage_report.json`'s
  1,427, so CF's compound count is slightly soft.
