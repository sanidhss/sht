# try13 — zero-shot base-model scoring: is prior familiarity why TTR fails?

**Status: submitted, result pending.** This file will be updated with the real numbers
when the job finishes.

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

Pending — job submitted 01-11-2026. Fill in with the measured table and state plainly
which of the three branches above the data actually took.
