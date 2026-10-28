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

8. **try09 (current)** — leak-free eval harness. Rebuilds the measurement rather than the
   model: hard negatives held out **by scaffold** (so near-copies cannot stand in for
   them), scaffold-split validation set with best-epoch selection (there was previously no
   validation set at all), a 500-decoy **enrichment percentile** replacing the unstable
   5-drug test, and 3 seeds reported as a distribution. Separates the two questions that
   were conflated in one pass/fail: *category* (enrichment) vs *within-family* (hard
   negatives) — the four targets fail at different levels. CF and TTR first, one job each.
   Job 22996488 (TTR) COMPLETED, 22996487 (CF) TIMED OUT after 1 of 3 seeds.
   **`best_epoch = 0` in all 4 completed runs** — the old 3-epoch pipeline, which had no
   validation set, trained ~3x past optimum, which alone explains the memorization. TTR
   goes from failing everything to **2/3 seeds passing** with seed spread of 41 points
   (vs 377 on the old pipeline); CF's leaky clean pass does **not** reproduce. See try09's
   Result.

9. **try10 (current)** — distribution-based scoring. Tests whether the output degeneracy
   that survived every try09 fix is a **readout artifact**: all scores to date came from
   `model.generate(...)` with no sampling argument, i.e. greedy decoding, which returns the
   *modal* output rather than an estimate. Compares greedy against a K=16 sampling mean and
   a digit-logit expectation on the same four saved adapters and try08's exact 200-molecule
   set, so the readout is the only variable. No training. Job 23130595 COMPLETED:
   **degeneracy was largely a readout artifact** — Gaucher's 791 falls from 51% of novel
   compounds to **5.5%** under digit-logit expectation, no retraining. But ranking did not
   uniformly improve, and **Fabry's apparent category-level success was a mode artifact**
   (15.5 → 83.0 percentile, worse than chance). See try10's Result.

## Current state

**Every pass/fail verdict recorded before try09 was produced by a leaky test AND an
over-trained model. Treat them all as void.** The evidence, in order of importance:

### 1. The pipeline trained ~3x too long, with no validation set at all
`best_epoch = 0` in all 4 completed try09 runs; validation loss rises essentially
monotonically after the first epoch while training loss keeps falling to ~0.19. The
original script ran a fixed 3 epochs and never evaluated held-out data. This is a mundane
and complete explanation for the memorization try08 measured — no architectural theory
needed. Fix: stop at epoch 0, and never train without a validation set again.

### 2. The hard-negative test leaked
All 12 hard negatives, across all four targets, were in their training sets; only the
reference was held out. So the test compared a generalization task against a memorization
task. This is what produced the "collapse" signature behind the Fabry and TTR diagnoses —
the model applying a memorized neighbour's label to a whole structural family. **Fabry's
stereochemistry/diastereomer diagnosis is most likely an artifact of this.**

### 3. Under the fixed harness, the target ranking inverts
| | old (leaky, over-trained) | try09 (leak-free, epoch 0) |
|---|---|---|
| **TTR** | fails everything; enrichment 47.7% (~chance); "worst target" | **2/3 seeds pass**; enrichment 19–25%; reference spread **41 points** |
| **CF** | clean pass; enrichment 11.5%; "only genuine success" | **fails on its 1 completed seed** (555 vs analogs 540/529); enrichment 21.4% |

On the category-level gate the two are now indistinguishable (CF 21.4% vs TTR 19–25%), so
the apparent CF/TTR gap was an artifact. **"CF good, TTR hopeless" does not survive.**
CF has only n=1 and must be re-run before "CF fails" is asserted.

### 4. Seed variance collapses once you stop at epoch 0
TTR's reference prediction spread across seeds is **41 points**, against the **377 points**
try07 measured on the old pipeline. Most of the instability was over-training, not the
task.

### 5. Output degeneracy was largely a greedy-decoding artifact (try10)
Every score this project ever produced came from `model.generate(...)` with no sampling
argument — greedy decoding, which returns the **modal** output rather than an estimate.
Switching to a digit-logit expectation (3 forward passes, deterministic, no retraining)
drops Gaucher's most-common value from **51% of novel compounds to 5.5%**, and raises
distinct values from 27.5% to 69%. Method A reproduced try08 at 100% on all four targets,
so this is the readout and not drift. **Digit-logit expectation is now the scoring method.**

But fixing the ties did **not** uniformly improve ranking, and one result is a serious
correction: **Fabry's apparent top-15.5% category performance was entirely an artifact of
its collapse value (436) happening to be a low number.** Under honest scoring migalastat's
expected value is 647–660, ranking **83rd percentile — worse than chance.** Fabry is the
weakest of the four targets, not the second strongest.

Honest ranking under digit-logit expectation (on the old over-trained adapters):
CF 14.0, Gaucher 29.0, TTR 45.0 (~chance), Fabry 83.0. **Every enrichment number recorded
before try10 was computed from modes and is suspect.**

Also: per-molecule sampling variance is not a quality proxy — TTR has the *lowest* variance
(26.8) and chance-level ranking, i.e. confidently wrong, while Fabry has the highest (82.2).

### 6. Ranking quality remains the open problem
TTR still assigns **994 to ~40% of a diverse 500-compound decoy set** (70–92 distinct
values out of 500); CF assigns 889 to 14%. So Gaucher's 791-on-51%-of-novel-compounds is
not merely an over-training artifact. This is the failure that most directly threatens the
downstream plan: a 40% tie-group makes any top-K shortlist arbitrary.

Related: validation loss at its best (2.26–2.97) is still far above training loss at the
same epoch (~1.0), so the model overfits *within* epoch 0. Epoch-level early stopping may
be too coarse — mid-epoch evaluation, lower LR, or fewer steps are the next things to try.

### Next steps, in order
1. **Re-run CF through the try09 harness with digit-logit scoring**, 3 seeds, `N_EPOCH=2`.
   try09's leak-free numbers were greedy-decoded too, so they need recomputing — and CF is
   still the only target with a plausible genuine pass. **Save the adapters this time**:
   try09 never called `save_pretrained`, which is why try10 had to run on the older
   over-trained weights. (epoch 0 is established as optimal across 4
   runs, so 5 epochs is wasted compute; this makes 3 seeds fit in budget).
2. **Re-run Gaucher and Fabry through the try09 harness.** Both of their recorded verdicts
   are void. Fabry's needs care: the scaffold holdout costs 20% of only 89 compounds.
3. **Attack output degeneracy** — the one problem none of the eval fixes touched.
4. **Then** revisit regularization and whether 5x SMILES augmentation still helps. Both
   were deferred deliberately so the measurement could be fixed first; augmentation's
   supporting evidence came from the broken test.
5. `NEXT_PHASE_diverse_structural_training.md` remains **pending re-justification**, now on
   a third count: the failures it was designed to address were substantially eval and
   over-training artifacts.

### Caveats on earlier numbers
- Gaucher's analog real labels are 791/**679**/839, not the 791/791/839 quoted in places
  (those are *predicted*). Analog #2's real 679 is more potent than ambroxol's real 685.
- CF's training file yields **1,450** distinct canonical compounds vs `coverage_report.json`'s
  1,427, so CF's compound count is slightly soft.
- `beat_easy` is computed, printed, saved and never used in the original script's verdict —
  which is how Gaucher printed "trust this target adapter for broader DrugBank-wide
  screening" while scoring worse than metformin. try09 sidesteps this by reporting
  enrichment separately; the original script still has the flaw.
