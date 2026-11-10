# try13 — zero-shot base-model scoring: is prior familiarity why TTR fails?

**Status: COMPLETE — job 23414025, 1h00m, 04-11-2026. This run found the cause.**

## The question this closes

try11 gave the project its first clean positive and its flattest null in the same run:

| target | drug | fine-tuned digit_exp enrichment (3 seeds) | verdict |
|---|---|---|---|
| gaucher | ambroxol | **0.2 / 0.2 / 0.8 percentile** | passes 3/3 leak-free |
| attr | tafamidis | ~50.8 percentile (815.7 vs decoy median 815) | no signal at all |

Four structural explanations for that gap were tested directly against the data. **All
four are dead**, and they die the same way: in every case, Gaucher succeeds with *less*
of whatever the explanation says TTR lacks.

| hypothesis | measurement | verdict |
|---|---|---|
| The model can't emit values as low as 552 | TTR's own labels span 0–1000; **33.7% are below 552** | ✗ dead |
| Tafamidis's chemical class is absent from training | True — 0 of 476 share its Murcko scaffold. But **0 of 1,202** share ambroxol's, and Gaucher works | ✗ dead |
| The scaffold holdout stripped its nearest neighbours | True — but ambroxol's best Tanimoto is **0.173** vs tafamidis's **0.340**. Less support, better result | ✗ dead |
| Not enough data (476 vs 1,202 compounds) | try07 downsampled Gaucher to **493** and it still passed 2/3 | ✗ dead |

What survives is the one explanation that does not live in the fine-tuning data at all:
**prior familiarity in the base model.** Ambroxol is a ubiquitous OTC mucolytic with
decades of literature; tafamidis is a 2011 orphan drug. If pretraining already encodes
ambroxol's activity profile, LoRA only has to avoid destroying it — whereas for
tafamidis it would have to construct the answer from 476 compounds whose closest
relative is a 0.34-Tanimoto stranger. That is the only account that explains the thing
none of the structural ones can: **why the drug with less structural support is the one
that works.**

The decisive property of this hypothesis is that it predicts an effect measurable
**with no training at all.**

## What this run does

One base-model load. No LoRA, no gradients, no optimizer, no epochs, no checkpoint to
select. None of the three bugs this project has fought — the training leak, the
over-training, the greedy-decoding artifact — can apply to a model that is never
modified.

Two independent readouts per molecule, on exactly try11's molecule sets:

**A. Zero-shot affinity.** The reference drug, its 3 hard negatives, and the *same* 500
MaxMin-diverse cross-target decoys (same builder, same `seed=0`), scored with try11's
exact dual readout — greedy and digit-logit expectation from one forward pass. Reported
as the same enrichment percentile try11 used, so the zero-shot and fine-tuned numbers
differ only by the adapter.

**B. SMILES familiarity.** Per-token negative log-likelihood of each molecule's own
SMILES string. This asks "how predictable is this structure to a model that has read the
literature," and it touches neither the prompt template, nor the protein, nor the digit
tokens. A molecule seen often in pretraining is cheaper to spell.

B exists because **A alone cannot distinguish "the base model knows ambroxol" from "the
base model knows neither drug."** If zero-shot affinity comes back flat for both, B is
the only axis left that can still separate them — and it is one forward pass per
molecule, cheaper than the generate call it rides along with. Each drug's NLL is reported
as a percentile against its own 500-decoy field, never as a raw number, because
per-token NLL depends on SMILES length and composition.

## How to read the result

Low affinity percentile = the drug ranks above the decoy field (good).
Low NLL percentile = the drug's structure is unusually familiar to the base model.

- **Ambroxol low, tafamidis high (on either axis)** → prior familiarity **is** the cause.
  TTR's failure is then not fixable with more TTR data; it needs a target where the base
  model already has a prior, or a different mechanism for supplying one.
- **Both flat on A and B** → familiarity is **not** the cause either. The difference was
  made by fine-tuning, and the search narrows to what training did differently between
  the two targets — a smaller and better-defined question than the one this run closes.
- **Tafamidis strong zero-shot, weak after fine-tuning** → fine-tuning **destroyed** a
  working prior. That makes TTR a training-damage problem, not a knowledge gap, and it
  would be the most actionable of the three.

All three are real answers. This run cannot come back uninformative, which is the point.

## Cost and scope

| | |
|---|---|
| job | `zeroshot_familiarity`, `gpuA100x4`, 1 GPU, 4h reserved |
| expected | **~2h** — 1 base load, 2,016 generate calls, 2,016 single-forward NLLs |
| basis | try08 measured **1h34m** for *four* base loads plus ~1,240 generations |
| writes | `zero_shot_familiarity/` only — no per-target directory, no adapter |

All four targets run in one job deliberately. The try05 scope-coupling bug was about a
script silently *training* on whichever targets a shared report marked ready; nothing is
trained here and every file is opened read-only, so that failure has no surface. Reusing
a single base-model load across four targets is the entire reason this is cheap. The
`TARGETS` env var still narrows it if needed.

`load_grouped` is memoized in this try (it is not in try11): the decoy builder reads every
*other* target's file, so running four targets in one process would otherwise re-parse and
re-canonicalize the same ~7,000 SMILES twenty times. The function is read-only and
deterministic, so this cannot change any result.

## Result

**Job 23414025, 1h00m, COMPLETED 04-11-2026.** Faster than the ~2h estimate.

### The measurement

| target | drug | NLL %ile (familiarity) | zero-shot digit %ile | zero-shot beats_hard | fine-tuned digit %ile | 
|---|---|---|---|---|---|
| gaucher | ambroxol | **5.4** | **18.4** | **True** | 0.2 / 0.2 / 0.8 (try11) |
| fabry | migalastat | **0.2** | 15.2 | False | 44.0 / 48.6 / 48.0 (try12) |
| cf | lumacaftor | 42.4 | 79.2 | False | 9.6 / 34.8 / 42.0 (try11) |
| attr | tafamidis | **70.2** | 74.6 | False | 57.0 / 50.8 / 84.8 (try11) |

Raw NLL against each drug's own 500-decoy field (decoy median ~10.0 for all four):
migalastat **2.83**, ambroxol **7.04**, lumacaftor 9.52, tafamidis **11.30**. Tafamidis is
the only reference drug the base model finds *harder* to spell than a typical decoy.

### The cause of TTR's failure

**Tafamidis is the only one of the four drugs that is both unfamiliar to the base model
and has no zero-shot signal.** It is last on familiarity (70.2 percentile, the only drug
above the decoy median) and last on zero-shot affinity (74.6 percentile — the decoy field
beats it). Ambroxol, the one target that passes, is the **only** drug in the entire
project that beats its hard negatives *before any training exists* — the single `True` in
the zero-shot column.

So Gaucher's result is **a pre-existing base-model prior that LoRA sharpened**
(18.4 → 0.2 percentile), not knowledge constructed out of 1,202 training compounds. TTR
has no prior to sharpen. Fine-tuning does produce real learning there — greedy moves 75.2
→ 26.3 mean, digit_exp 74.6 → 64.2 mean — but starting from nothing, 476 compounds whose
nearest relative is a 0.34-Tanimoto stranger only get it to roughly chance.

**This resolves the paradox that killed all four structural hypotheses.** Each of them
asked "what does TTR's *training data* lack that Gaucher's has," and each died because
Gaucher has less of it. They died because the question was wrong: the knowledge Gaucher
uses is not in its fine-tuning data at all. It is in pretraining. Nothing measured inside
the fine-tuning data could ever have separated these two targets, which is exactly what
four dead hypotheses look like in hindsight.

### The other branch also fired: Fabry is training damage

Migalastat is the **most** familiar molecule of the four by a wide margin (NLL 2.83 vs a
decoy median of 10.14 — percentile 0.2, meaning 499 of 500 decoys are less familiar) and
it scores 15.2 percentile zero-shot. Every fine-tuned Fabry number is **worse**: 44–49
(try12) and 83 (try11). **Zero-shot is the best result Fabry has ever produced.**

And the diastereomer collapse is a **base-model property, not a data-volume artifact**.
Zero-shot greedy returns **331 for migalastat and all three diastereomers**, identically,
with real labels of 196 / 925 / 754 / 993. digit_exp separates them by under 7 points.
This happens with zero training compounds involved, which retires the recorded root cause
("only 89 real training compounds") for good.

### Why familiarity alone is not the predictor

Migalastat is the most familiar drug and fails; lumacaftor is middling on familiarity
(42.4) and fine-tunes to the second-best result (9.6 on its best seed). So familiarity is
**necessary context, not a sufficient test.** What sorts the four targets correctly is
**zero-shot hard-negative discrimination** — whether the base model can already tell the
drug from its closest real analogs. Exactly one target can, and exactly one target passes.

### What this is good for

The practical finding is bigger than TTR. **Whether this pipeline will work on a given
orphan drug is measurable up front, in one hour, with no training:** score the drug and
its analogs zero-shot and check whether it separates. For a thesis about rare diseases
generally, that is a usable screening criterion rather than a per-disease post-mortem —
it says which orphan targets are worth fine-tuning before spending the GPU time, and it
predicted all four outcomes here.

### Caveat

Fabry's enrichment is over 499 decoys, not 500 — one decoy returned no parseable score.
Immaterial to a percentile. Everything else ran clean on all four targets.
