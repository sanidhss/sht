# Project: TxGemma-Predict (numeric drug-target scoring)

**Question:** Can `txgemma-9b-predict`'s `BindingDB_ki`-style numeric scoring correctly
rank a known drug (migalastat/GLA, ambroxol/GBA1) above unrelated and structurally
near-identical decoys, for orphan disease targets?

## Current state (25-11-2026)

17 tries. Two things in this project are defensible results and the rest is
instrumentation:

1. **The up-front screening criterion (try13).** Whether this pipeline will work on a
   given orphan drug is measurable in one GPU-hour with no training: score the drug and
   its closest real analogs zero-shot and check whether the base model separates them.
   It sorted all four original targets correctly. This is the transferable finding.
2. **Fine-tuning converges every target to the same band (try14/try15, 8 targets).**
   Effects up to 80 percentile points in both directions — cel 94.2 → 20.6 and
   nsiad 96.6 → 16.0, but fabry 10.0 → 35.2 and mastocytosis 14.6 → 77.9 are *damaged*.
   LoRA teaches the task's output distribution more than the target's chemistry.

**Everything numeric here is pool-dependent and most of it is superseded.** All pre-try09
pass/fail verdicts are void (leaky hard negatives + ~3x over-training); all pre-try10
enrichment numbers were computed from greedy modes; and try15 showed percentiles move by
up to 31 points when only the decoy pool changes. **try16/try17 move this arm onto the
frozen 1,933-drug equalized pool** (`06_equalized_pool/try01`), which is the only field
comparable with the TxGNN arm. 2 of 6 targets have landed — see section 12.

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

### 7. THE CAUSE, found 04-11-2026 (try13): success tracks the *base model's* prior, not the training data

This is the most important finding in the project so far, and it reframes everything
above it. All four targets were scored on the **untouched base model, zero-shot** — no
LoRA, no gradients, nothing learned:

| target | drug | SMILES NLL %ile (familiarity) | zero-shot digit %ile | zero-shot beats_hard | fine-tuned digit %ile |
|---|---|---|---|---|---|
| gaucher | ambroxol | **5.4** | **18.4** | **True** | 0.2 / 0.2 / 0.8 |
| fabry | migalastat | **0.2** | 15.2 | False | 44.0 / 48.6 / 48.0 |
| cf | lumacaftor | 42.4 | 79.2 | False | 9.6 / 34.8 / 42.0 |
| attr | tafamidis | **70.2** | 74.6 | False | 57.0 / 50.8 / 84.8 |

**Ambroxol is the only drug in this project that beats its hard negatives before any
training exists** — the single `True` in that column, and the only target that passes.
Gaucher's result is therefore a **pre-existing base-model prior that LoRA sharpened**
(18.4 → 0.2 percentile), not knowledge built from 1,208 training compounds.

**TTR's cause:** tafamidis is the only reference drug that is both unfamiliar to the base
model (NLL 70.2 percentile — the only one above the decoy median, raw NLL 11.30 vs
ambroxol's 7.04) and has no zero-shot signal (74.6 percentile). There is no prior to
sharpen. Fine-tuning does produce real learning — greedy 75.2 → 26.3 mean, digit_exp 74.6
→ 64.2 mean — but from a cold start, 476 compounds whose nearest relative is a
0.34-Tanimoto stranger only reach chance.

**Why four structural hypotheses all died.** Each asked "what does TTR's *training data*
lack that Gaucher's has," and each died because Gaucher has *less* of it (label range,
scaffold presence, nearest-neighbour support, compound count — see
`try11_30-10-2026_step-level-stopping/README.md`). They died because the question was
wrong. The knowledge Gaucher uses is not in its fine-tuning data; it is in pretraining.
Nothing measurable inside the fine-tuning data could have separated these two targets.

**Fabry is training damage, not a knowledge gap.** Migalastat is the *most* familiar
molecule of the four by a wide margin (NLL 2.83 vs decoy median 10.14 — 499 of 500 decoys
are less familiar) and scores 15.2 percentile zero-shot. Every fine-tuned Fabry number is
worse: 44–49 (try12) and 83 (try11). **Zero-shot is the best result Fabry has ever
produced.** The diastereomer collapse is also a base-model property, not a data-volume
artifact: zero-shot greedy returns **331 for migalastat and all three diastereomers**,
identically, with real labels 196/925/754/993. This retires the long-recorded root cause
("only 89 real training compounds") permanently.

**Familiarity alone is not the predictor** — migalastat is the most familiar and fails.
What sorts all four targets correctly is **zero-shot hard-negative discrimination**:
whether the base model can already tell the drug from its closest real analogs.

**The usable finding for rare diseases generally:** whether this pipeline will work on a
given orphan drug is measurable up front, **in one GPU-hour, with no training.** Score
the drug and its analogs zero-shot and check whether it separates. That is a screening
criterion applicable to any orphan target, and it predicted all four outcomes here —
which is a far more transferable result than a per-disease post-mortem.

### 8. CF and Fabry, final numbers (try11 job 23400943, try12 job 23400706)
- **CF: 2/3 seeds beat hard negatives on both readouts, but enrichment is 9.6 / 34.8 /
  42.0 (spread 32.4).** Real learning happened — zero-shot was 79.2 — but the spread is
  too wide to call a pass. The selection check **FAILED for 2 of 3 seeds** (full
  all-variants validation ranked the runner-up checkpoint better), so even CF's 228–254
  distinct validation molecules are too few to resolve its curve. Val minimum is at step
  204–312, always inside epoch 0.
- **Fabry: cross-validation fixed the instrument and the answer is a clean negative.**
  6-fold scaffold CV put the optimum at 0.420 epochs with a clean interior minimum; 3
  seeds at that point give digit_exp 44.0 / 48.6 / 48.0 — **spread collapsed from 38.4 to
  4.6.** The try11 inconclusiveness *was* the 11-molecule checking set. What the
  trustworthy measurement says is chance.

### Next steps, in order
1. **Re-frame the thesis claim around what try13 found.** The defensible result is not
   "TxGemma-predict ranks chaperones for orphan targets" — it is "whether it can is
   predictable before training, from a one-hour zero-shot measurement, and we have the
   criterion that sorted four targets correctly." That is a positive, transferable finding
   about rare diseases generally, which was the stated requirement.
2. **Decide whether fine-tuning should be kept at all for targets with a strong prior.**
   Fabry is strictly better zero-shot, and Gaucher's gain (18.4 → 0.2) is the only case
   where LoRA clearly helps. Worth testing zero-shot against fine-tuned on the full
   3,417-drug approved library (`/work/hdd/bisa/sshah21/results/reasoning_screen/chembl_approved.json`,
   built by the chat track) rather than 500 decoys.
3. **CF needs a wider validation set before its numbers mean anything** — its selection
   check failed on 2 of 3 seeds. Cross-validation, as done for Fabry in try12, is the
   known fix.
4. `NEXT_PHASE_diverse_structural_training.md` is now **superseded, not just pending**:
   it was designed to fix structural discrimination in the fine-tuning data, and try13
   shows the fine-tuning data is not where the discrimination comes from.

### 9. New targets vs the original four (try14, partial — 12-11-2026)

Two of try14's four runnable targets have finished (`cel`, `tyrosinemia`). The same job
also ran the rebuilt RDKit descriptor baseline over **all eight** targets, so for the
first time the original four have a non-LLM comparison in identical units.

**Enrichment percentile, digit_exp readout, mean over 3 seeds (lower = better, 50 = chance):**

| target | try | n_train | model | kNN | ridge | best baseline | model wins? |
|---|---|---|---|---|---|---|---|
| gaucher | 11 | ~1010 | **0.4** | 21.6 | 65.6 | 21.6 | **yes, decisively** |
| cf | 11 | ~1190 | 28.8 | 52.6 | 29.4 | 29.4 | tie (28.8 vs 29.4) |
| fabry | 11 | 60 | 35.2 | 26.4 | 97.0 | 26.4 | no |
| attr | 11 | ~390 | 64.2 | 31.0 | 42.8 | 31.0 | no |
| cel | 14 | ~1260 | 20.6 | **5.8** | 16.6 | 5.8 | no |
| tyrosinemia | 14 | ~135 | 35.3 | 26.8 | 45.2 | 26.8 | no |

**The new targets did not perform worse than the old ones.** cel's 20.6 is the
second-best enrichment in the project, better than CF, Fabry and TTR. What changed is
that the baseline is now in the table, and against it the model wins once — Gaucher,
which try13 showed is a base-model prior LoRA sharpens rather than something learned
from the training data. Everything else is a loss or a tie. **The original four were
already losing to a Tanimoto k-NN; nobody had measured it.**

**More data did not help.** cel has the most training compounds of any target in the
project (~1,260, above Gaucher's ~1,010) and lands 50x worse than Gaucher on enrichment.
This is consistent with try07's downsampling result and with try13: data volume is not
the lever.

#### The hard-negative test does not mean the same thing on both sets

Gaucher and CF pass `beats_hard` 3/3; both try14 targets fail 0/3. That gap is not a
performance difference. Reading the eval sets shows the test is constructed differently
because the targets are different, in two ways.

**1. The original four's hard negatives are genuinely distinct molecules. Three of
try14's are the reference drug.** Hard negatives are the top-3 Tanimoto neighbours of
the reference with no identity check, and similarity to a salt, tautomer or
stereo-unannotated copy is ~1.0:

| set | target | hard negatives are |
|---|---|---|
| original | gaucher | 3 unrelated brominated scaffolds |
| original | cf | 3 distinct thiazole/pyridine analogs |
| original | fabry | 3 true diastereomers of migalastat |
| original | attr | 3 real xanthone analogs |
| try14 | cel | **imatinib mesylate** (CHEMBL1642 — the reference plus a counterion) + 2 real analogs |
| try14 | tyrosinemia | **both enol tautomers of nitisinone** + 1 real analog |
| try14 | mastocytosis | **staurosporine twice** (with and without stereo annotation, identical 14.7 nM label) + 1 stereoisomer |

ChEMBL lists salt and tautomer forms as separate molecule entries with their own
activity records. Deduplication is by canonical SMILES, which does not merge them. The
original four's reference drugs happen to appear once each; imatinib and nitisinone do
not, because they have far larger medicinal-chemistry literatures.

**2. On the original four the drug is the most potent molecule in its neighbourhood. On
the new oncology targets it is not — by design.**

```
original:  fabry   migalastat  40.4 nM  vs  390000 / 45000 / 920000 nM   clean
           attr    tafamidis    839 nM  vs   13000 / 10000 /  12000 nM   clean
           gaucher ambroxol   10979 nM  vs   50000 / 10000 / 100000 nM   one near-tie
           cf      lumacaftor  1293 nM  vs    5385 /  1294 /   3990 nM   one exact tie

try14:     mastocytosis midostaurin 57.2 nM  vs  14.7 / 14.7 / 0.2 nM    ALL MORE POTENT
           nsiad        tolvaptan    3.4 nM  vs  1200 /  8.0 / 3.0 nM    one inversion
```

Midostaurin is the approved drug for selectivity and ADME, not raw KIT potency;
staurosporine is the more potent pan-kinase inhibitor and is undruggable for exactly
that reason. **A correct model must fail mastocytosis's `beats_hard`.**

This is the structural point, and it limits the thesis claim: the hard-negative test as
designed only poses a well-formed question for diseases where potency *is* the whole
story — chaperones and stabilisers, i.e. the original four. For oncology kinase
inhibitors, "rank the drug above its close analogs" is the wrong question, because the
drug was never selected for being the best binder.

#### What got better

The step-selection audit passed **6/6 seeds** on try14 against **4/6** in try11 (CF
failed 2 of 3, TTR 1 of 3). The new targets have more validation molecules, so the
stopping point is better determined. CF's instability is CF's, not the method's.

Greedy's failure mode is also now unambiguous: on cel it produced 54–57 distinct values
across 500 decoys (~89% collapse) against digit_exp's 239–253, with a cross-seed spread
of 47.6 vs 4.6. Report digit_exp.

### 10. The corrected picture, all eight targets (try14 + try15, 22-11-2026)

Supersedes the reading in section 9, which was written on 2 of 8 targets and on a
zero-shot control that had only ever been run on the original four.

**Everything below uses the 8-target decoy pool.** Section 11 explains why that matters.

| target | zero-shot | fine-tuned | best baseline | fine-tuning did | vs baseline |
|---|---|---|---|---|---|
| fabry | 10.0 | 35.2 | 26.4 | **damaged** | baseline |
| gaucher | 11.0 | **0.4** | 21.6 | sharpened | **model** |
| mastocytosis | 14.6 | 77.9 | 20.8 | **destroyed** | baseline |
| cf | 47.8 | 28.8 | 29.4 | learned | tie |
| attr | 58.0 | 64.2 | 31.0 | nothing | baseline |
| tyrosinemia | 86.6 | 35.3 | 26.8 | learned a lot | baseline |
| cel | 94.2 | 20.6 | 5.8 | learned a lot | baseline |
| nsiad | 96.6 | 16.0 | 9.2 | learned a lot | baseline (2/3 seeds win) |

**Fine-tuning produces the largest effects in this project — up to 80 percentile points.**
The earlier conclusion that it adds nothing was drawn from four targets, two of which
happened to have strong base-model priors, and it does not survive contact with the new
four. cel and nsiad start *actively wrong* (94.2, 96.6 — the real drug ranked among the
least potent of 501) and end at 20.6 and 16.0.

**But it converges everything to the same band.** Nothing finishes below ~16 except
Gaucher; nothing finishes above ~65 except mastocytosis. Targets starting at 87–97 are
pulled down to 16–35; targets starting at 10–15 are pushed up to 35–78. The honest
description is that **LoRA is teaching the task's output distribution more than the
target's chemistry** — which simultaneously explains the gains and the damage.

**Damage is a repeatable failure mode, not a Fabry quirk.** Of the three targets whose
base model had a good prior, two were degraded: fabry 10.0 → 35.2 and mastocytosis
14.6 → 77.9. Only Gaucher improved from a good start.

**Against the bar: 1 win, 1 tie, 6 losses.** Every target fine-tuning rescued, a Tanimoto
k-NN had already solved better. That verdict is unchanged from section 9 and is the one
robust result.

**Zero-shot screening does not generalise.** Only gaucher passes `beats_hard` zero-shot
legitimately. Mastocytosis also passes, but its analogs are genuinely more potent than
midostaurin, so that pass is a factual error the flag records as success. n = 1 over
eight targets.

### 11. Enrichment percentiles are decoy-pool-dependent (try15)

`build_decoys()` pools every other target in `ALL_TARGETS`, so that list defines the
yardstick. Re-scoring the original four zero-shot against an 8-target pool instead of
try13's 4-target pool moved every number, one by 31 points:

| target | try13 (4-pool) | try15 (8-pool) | shift |
|---|---|---|---|
| gaucher | 18.4 | 11.0 | −7.4 |
| cf | 79.2 | 47.8 | **−31.4** |
| fabry | 15.2 | 10.0 | −5.2 |
| attr | 74.6 | 58.0 | −16.6 |

Same model, same drugs, same prompts. Only the comparison set changed. A wider pool holds
more molecules the model rates as potent, lifting the reference's rank.

**No percentile in this project is comparable across tries unless both used the same
`ALL_TARGETS`.** The ordering is stable under both pools, so try13's conclusions hold and
its numbers do not transfer. Every table in the thesis must state its decoy pool. In
particular, try11's fine-tuned numbers for the original four are 4-pool and cannot be
differenced against try15's 8-pool zero-shot — which is why the table in section 10 shows
direction for those four rather than a delta.

### Caveats on earlier numbers
- Gaucher's analog real labels are 791/**679**/839, not the 791/791/839 quoted in places
  (those are *predicted*). Analog #2's real 679 is more potent than ambroxol's real 685.
- CF's training file yields **1,450** distinct canonical compounds vs `coverage_report.json`'s
  1,427, so CF's compound count is slightly soft.
- The recorded Fabry root cause ("only 89 real training compounds") is **wrong** — see
  section 7. The diastereomer collapse exists in the base model with zero training
  compounds involved, and try07 showed Gaucher passes at 493.
- `beat_easy` is computed, printed, saved and never used in the original script's verdict —
  which is how Gaucher printed "trust this target adapter for broader DrugBank-wide
  screening" while scoring worse than metformin. try09 sidesteps this by reporting
  enrichment separately; the original script still has the flaw.

### 12. On the frozen equalized pool (try16/try17, partial — 25-11-2026)

try15's lesson was that a percentile is a statement about a candidate set. try16 and
try17 therefore drop the 500-decoy field entirely and score
`06_equalized_pool/try01`'s **pool_v1: 1,933 real approved drugs, sha256-verified inside
every scoring script**, which is the same field the TxGNN arm re-ranked onto in try25.
The question becomes the deployment question: how deep a shortlist of approved drugs must
a screen carry before the right drug is in it.

**Scope: 6 targets, not 8.** Ambroxol and tolvaptan are absent from pool_v1 by canonical
structure (ambroxol is not `max_phase=4` in ChEMBL; tolvaptan's UniChem partner is a
different ChEMBL record from the library's). **Nothing was substituted** — tolvaptan's
UniChem partner CHEMBL4303142 is the same record try14 used as nsiad's *hard negative*,
a 350x weaker stereoisomer. So **Gaucher, this project's one success, cannot be placed on
the shared ruler at all.** That is a limitation of the pool, and it lands squarely on the
thesis's stated n=2 (Fabry + Gaucher) design.

**Status: 2 of 6 targets scored.** try16 array tasks 0/1 (`cf`, `fabry`) completed 25-11
after ~37h queued; tasks 2-5 (`attr`, `cel`, `mastocytosis`, `tyrosinemia`) and all of
try17 are PENDING on priority. The `SUMMARY.md` in try16's results directory is the 23-11
dry-run collate and still says all eight are missing — it is not the current state.

| target | model blinded (digit_exp) | seed spread | Tanimoto blinded | chance |
|---|---|---|---|---|
| cf / lumacaftor | **38.89%** (14.5 / 43.8 / 58.4) | **43.8** | 71.03% | 49.90% |
| fabry / migalastat, crossval | 40.49% | **4.4** | 85.83% **TRIVIAL** | 49.90% |
| fabry / migalastat, stepstop | 27.09% | 29.5 | 85.83% **TRIVIAL** | 49.90% |

Four things follow, all detailed in try16's README:

- **The 1-win/1-tie/6-loss verdict against the Tanimoto baseline does not survive the
  change of field, at least on cf.** All three cf seeds beat 71.03%; the decoy field had
  been flattering that baseline by 41.6 points. But three of the four pending targets
  carry the baselines that matter most (mastocytosis 0.57%, cel 17.80%, tyrosinemia
  18.42%), so the verdict is open, not overturned.
- **fabry has no admissible baseline here.** Its Tanimoto row is `trivial_flag = True`
  (a training active at Tanimoto 1.000 — Morgan-2 is stereo-blind, so migalastat's
  diastereomers are *identical* to it in that representation, with labels 925/754/993
  against migalastat's 196). Beating 85.83% is not evidence of anything.
- **Cross-validation stabilises fabry again** — spread 4.4 vs stepstop's 29.5, a 6.7x
  reduction reproducing try12 on a new field. Report crossval (40.49%, just better than
  chance), not stepstop's lower but noisier 27.09%.
- **`abs_label_error` separates the two targets in a way the ranks hide.** cf's
  prediction is within 3-7% of lumacaftor's measured 584 on every seed. fabry's is off by
  ~3x (524-605 against a real 196) on all six adapters, in the same direction every time.
  fabry's mid-field rank is a confident wrong answer about the most potent reference in
  the project, not a near-miss on a weak binder.

**No hard-negative result is reported in either try**, carried forward from try14: the
selection takes the top-3 Tanimoto neighbours with no identity or potency check, so on 3
of 4 new targets it selected the reference drug in disguise, and on mastocytosis the
negatives are genuinely more potent than midostaurin, so a correct model must fail.

try17 adds the unblinded ceiling (the reference injected with its own measured label),
and its two guard fixes are worth recording even before the GPU arm lands: the scaffold
holdout would have **deleted** the injected reference on fabry and cel, and the scaffold
validation split would have put it in validation on 4 of 18 seed-runs. **Without both
fixes "unblinded" would have been false on half the targets.** The CPU arms are already
in, and Finding 1 there is that unblinding does *not* let a 5-NN ace the task — k=5
averaging dilutes the exact match, and the rank is set by the other 1,932 drugs' scores,
not the reference's.
