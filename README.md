# Can AI find new uses for existing drugs in rare diseases?

A senior honors thesis. This page is written to be readable without a computing
background; every technical term is defined the first time it appears.

## The problem

Many rare ("orphan") diseases are caused by a single protein that folds into the wrong
shape. Some existing drugs work by grabbing onto such a protein and holding it in the
right shape. Finding more drugs like that is slow and expensive.

The question here: **can AI models pick the right drug out of a large list, without
having been told the answer?**

To test this, we use diseases where the answer is already known. Fabry disease is
treated by migalastat; malignant hyperthermia by dantrolene; cystic fibrosis by
lumacaftor. We hide those answers from the model, give it a list of 1,933 approved
drugs, and ask it to rank them. Then we check where the real drug landed.

If the model is working, the right drug comes out near the top. If the model is
guessing, it lands around the middle — position ~967 of 1,933.

## How to read the numbers

Everything in this project is reported as a **percentile**: the drug's position in the
ranking, as a percentage. **Lower is better.**

- **0.6%** means the right drug came 11th out of 1,933. Excellent.
- **50%** means it came out around 967th. That is exactly what random guessing gives.
- **85%** means the model ranked it worse than most random drugs.

Random chance is **49.9%**. Any number near 50 means nothing was found.

## Three approaches, deliberately kept separate

Three different kinds of AI are tested independently, because each one looks at a
completely different kind of evidence. If two of them agree, that means something; if
they disagree, that tells us where to look.

| approach | what it looks at | what it never sees |
|---|---|---|
| **TxGNN** (a "knowledge graph") | a giant map of which diseases connect to which genes and drugs | the chemical structure of any drug |
| **TxGemma-predict** (a language model) | the chemical structure of drugs, plus lab measurements of how tightly they stick to proteins | the disease map |
| **TxGemma-chat** (a reasoning model) | written descriptions of how drugs work, in plain English | either of the above |

## The four tests a result has to survive

This is the heart of the project. A good-looking number is easy to produce by accident,
and most of this work has been about catching those accidents.

**1. Blinding.** The model must never be shown the answer during training. Sounds
obvious; was violated repeatedly in early attempts, each time producing an impressive
result that later evaporated.

**2. The same list of drugs for everyone (the "equalized pool").** A model looks much
better when the wrong answers are obviously wrong. Early versions ranked the real drug
against 500 unrelated compounds, which is too easy. Now every approach ranks the *same*
frozen list of 1,933 approved drugs, so the scores can actually be compared. Changing
this list alone moved scores by 31 percentage points, which is larger than most of the
differences we were trying to measure.

**3. A dumb baseline to beat (the "Tanimoto 5-NN").** Before crediting an AI, check
whether something trivial does the same job. The trivial method here is pure chemical
look-alike matching: take compounds already known to stick to the target protein, and
rank the 1,933 drugs by how similar they *look* chemically. No AI, no training, runs in
seconds. If that beats the model, the model has added nothing.

It does beat both models on two diseases. On systemic mastocytosis it puts the right
drug 11th while both AI models sit at chance.

**4. The specificity check (for TxGNN only).** TxGNN builds a separate model for each
disease. So we can ask a *wrong* disease's model about a drug it should know nothing
about.

The result is uncomfortable. Tafamidis is the drug for TTR amyloidosis. TTR's own model
ranks it 155th — but the **Fabry** model ranks it **8th**. The wrong model does better.

The reason: the Fabry model gives a high rank to almost any approved rare-disease drug
you hand it (median position 96 out of 1,933), while the TTR model ranks them all near
chance (median 986). Neither is tracking its own disease. One says yes to everything in
that class, the other says no to everything.

A real hit looks different. The malignant hyperthermia model ranks those same drugs at
median 1,370 — it says no to nearly everything — **except** dantrolene, the correct
answer, at 11th.

## Where things stand

**TxGNN (the disease map).** Thirteen disease targets tested. Three pass the specificity
check — malignant hyperthermia, Gaucher, and phenylketonuria. Of those, PKU is dropped
for a separate reason: its disease entry is too sparsely connected for its score to be
trustworthy. So:

- **Malignant hyperthermia / dantrolene — 11th of 1,933, and ranked 458th or worse by
  every unrelated disease model.** The one unambiguous result in the project.
- **Gaucher / miglustat — 10th at its own disease, 27th under the Fabry model.** Passes,
  but the margin is thin, and part of the signal is borrowed from a neighbouring disease
  that also lists miglustat.
- **PKU / sapropterin — 211th vs 778th elsewhere.** Specific, but 211 of 1,933 is not a
  find, and the underlying data is too thin to rely on.
- Everything else fails.

An earlier headline result — Fabry in the top 0.2% — is **void**. The same disease
turned out to be present in the data a second time under a different name, carrying all
23 of its genes and the answer along with it. The model wasn't predicting; it was reading
the answer.

**TxGemma-predict (the chemistry model).** Eight targets. The useful finding is not a
per-disease win or loss but a **rule for deciding when to bother training at all**:
fine-tuning pulls every target toward the same middling band no matter where it started.
A target the base model already handles well is usually made *worse* by training; a
target it handles badly is usually improved. Since you can measure the starting point in
about an hour of computing time, you can predict the outcome before spending the effort.

On the frozen 1,933-drug list, all six testable targets finished on 25-11: two genuine
wins over the dumb baseline, one apparent win that is an artifact, and three losses —
the worst being systemic mastocytosis, where the model scores 63% and plain chemical
similarity scores 0.6%.

**TxGemma-chat (the reasoning model).** Removed from the main comparison. Kept for one
finding that is interesting on its own: changing nothing but the *wording* of the
question — same disease, same model, same drug list — moved the correct drug from 7th to
1,675th out of 1,933. A 239-fold swing from phrasing alone. A simple rule-based chemical
filter beat the model on all four tests.

**Structural work.** Not started. The jobs are queued.

## The honest record

Every apparent success in this project turned out to be a measurement error until the
measurement was rebuilt. None was caught by looking at the results and sensing something
was off — each was caught by a control that had been set up in advance specifically to
catch it. That track record is the most transferable thing here.

| what went wrong | found by | what it had faked |
|---|---|---|
| the answer was in the training data | 01/try08 | every result before that point |
| the model was trained ~3x too long with nothing to check against | 01/try09 | the signature of memorization |
| the scoring method returned the most common answer, not an estimate | 01/try10 | every enrichment number so far |
| scores depended on which wrong answers were in the list (31-point swings) | 01/try15 | all comparisons between attempts |
| the same disease appeared twice under a different name | 03/try18 | Fabry's top-0.2% result |
| ranking by disease *category* mistaken for ranking by disease | 03/try17, 03/try25 | Fabry *and* Gaucher |
| guessing which data was held out instead of checking | 03/try26 | would have mis-scored 14% of cases |
| "hard" decoys that were actually salts of the right answer | 01/try14 | 3 of 4 new targets' verdicts |

## What is still missing

- **The ceiling tests are queued, not finished.** We know where each approach lands, but
  not how well it could possibly do if unblinded. Both are waiting on the cluster.
- **Gaucher cannot be placed on the shared list at all.** Its drug, ambroxol, is not
  registered as fully approved in the chemical database used to build the list. This
  matters because Fabry + Gaucher was the thesis's stated proof-of-concept pair. It needs
  a decision: drop Gaucher, or rebuild the list with a hand-curated exception and report
  it as such.
- **The side-by-side comparison figure is kept off this repository** by deliberate
  decision. Each approach's own numbers stay in its own section.

## Decisions already made, not to be relitigated

- **Do not add chemistry to TxGNN.** The whole point is that it uses *only* the disease
  map. Blurring that destroys the comparison, and it is a large, risky change.
- **Do not keep building Fabry-specific fixes.** That teaches a narrow trick for one
  family of enzymes with no evidence it carries over, which is precisely what the thesis
  needs to establish. A later test showed the specific failure being chased is a property
  of the base model itself, present with no training data involved at all — so there was
  never a fix available on the data side.

## How this repository is organized

Work is grouped by project, then by attempt. Each attempt lives in its own dated folder
(`tryNN_DD-MM-YYYY_short-description`) containing the code as it stood at the time, the
real logs from the computing cluster, and a write-up of what was tried and — when it
failed — why.

| folder | what it is |
|---|---|
| [01_txgemma_predict](01_txgemma_predict/README.md) | the chemistry model (17 attempts, 8 targets) |
| [02_txgemma_chat](02_txgemma_chat/README.md) | the reasoning model (12 attempts, descoped) |
| [03_txgnn](03_txgnn/README.md) | the disease map model (26 attempts, the longest debugging effort) |
| [04_setup](04_setup/README.md) | one-time environment and data setup |
| [05_multi_agent](05_multi_agent/README.md) | source of the dumb chemical-similarity baseline |
| [06_equalized_pool](06_equalized_pool/try01_23-11-2026_build-pool/README.md) | builds and freezes the shared 1,933-drug list |
| [07_structure_pipeline](07_structure_pipeline/README.md) | the structural half of the thesis (not started) |

Large files — model weights, the disease database, cached downloads — are **not** stored
here. They stay in their original location on the cluster. This repository holds code,
logs and write-ups only.

### A note on the shared drug list

The 1,933-drug list is frozen and fingerprinted (a checksum beginning `9d89c3ce`), and
every scoring script verifies that fingerprint before running and refuses to start if it
does not match. This is deliberate: the single largest source of fake results in this
project was comparing numbers that had been measured against different lists of drugs.
