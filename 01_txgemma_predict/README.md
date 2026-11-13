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
