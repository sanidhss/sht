# Can AI find new uses for existing drugs in rare diseases?

A senior honors thesis. Written to be readable without a computing background.

## The problem

Many rare diseases are caused by one protein folding into the wrong shape. Some drugs
work by holding such a protein in the right shape. Finding more of them is slow.

**Can an AI model pick the right drug out of a large list, without being told the
answer?**

We test on diseases where the answer is already known — Fabry disease is treated by
migalastat, malignant hyperthermia by dantrolene. We hide the answer, hand the model
1,933 approved drugs, and check where the real one lands.

## How to read the numbers

Everything is a **percentile** — position in the ranking, as a percentage. **Lower is
better.**

| score | meaning |
|---|---|
| 0.6% | 11th of 1,933. Excellent. |
| 49.9% | exactly what random guessing gives |
| 85% | worse than most random drugs |

## Three approaches, kept separate

Each looks at a completely different kind of evidence, so agreement between them means
something.

| approach | sees | never sees |
|---|---|---|
| **TxGNN** | a map of which diseases connect to which genes and drugs | any drug's chemical structure |
| **TxGemma-predict** | chemical structures and lab binding measurements | the disease map |
| **TxGemma-chat** | plain-English descriptions of how drugs work | either of the above |

## The four tests a result has to survive

Most of this project has been about catching good-looking numbers that turned out to be
accidents.

**1. Blinding.** The model must never see the answer during training. Obvious, and
violated repeatedly in early attempts — each time producing a result that later
evaporated.

**2. One shared drug list.** A model looks better when the wrong answers are obviously
wrong. Early versions ranked against 500 unrelated compounds, which is too easy. Now
every approach ranks the same frozen list of 1,933 approved drugs. Changing this list
alone moved scores by 31 percentage points — more than most differences we were trying to
measure.

**3. A trivial baseline to beat.** Take compounds already known to stick to the target
protein, and rank the 1,933 by chemical resemblance to them. No AI, no training, seconds
to run. If that wins, the model added nothing.

It beats both models on two diseases. On systemic mastocytosis it puts the right drug
11th while both AI models sit at chance.

**4. The specificity check** (TxGNN only, which builds one model per disease). Ask the
*wrong* disease's model about a drug.

Tafamidis treats TTR amyloidosis. TTR's own model ranks it 155th; the **Fabry** model
ranks it **8th**. The wrong model wins — because the Fabry model rates almost any
approved rare-disease drug highly (median position 96 of 1,933) while TTR's rates them
all at chance (median 986). Neither responds to its own disease.

Malignant hyperthermia shows what a real result looks like: median 1,370 — it rejects
nearly everything — **except** dantrolene at 11th.

## Where things stand

**TxGNN (disease map).** Thirteen targets. Three pass specificity; one of those, PKU, is
dropped because its disease entry is too sparsely connected to trust.

- **Malignant hyperthermia / dantrolene — 11th of 1,933, and 458th or worse under every
  unrelated disease.** The one unambiguous result in the project.
- **Gaucher / miglustat — 10th at its own disease, 27th under Fabry's.** Thin margin, and
  part of the signal is borrowed from a neighbouring disease that also lists miglustat.
- **PKU / sapropterin — 211th vs 778th.** Specific, but not a find, and the data is thin.
- Everything else fails.

An earlier headline — Fabry in the top 0.2% — is **void**. The same disease appeared
twice in the data under different names, carrying all 23 genes and the answer with it.

**TxGemma-predict (chemistry).** Eight targets. The useful finding is a **rule for
deciding when to bother training**: fine-tuning pulls every target toward the same
middling result, so a target the base model already handles well usually gets *worse*.
The starting point is measurable in about an hour, so the decision can be made in advance.

On the shared list, all six testable targets finished 25-11: two genuine wins over the
trivial baseline, one artifact, three losses. The worst is mastocytosis — model 63%,
chemical similarity 0.6%.

**TxGemma-chat (reasoning).** Removed from the comparison. Kept for one finding: changing
only the *wording* of the question moved the correct drug from 7th to 1,675th of 1,933. A
239-fold swing from phrasing alone. A simple rule-based filter beat it on all four tests.

**Structural work.** Not started; jobs queued.

## The honest record

Every apparent success here turned out to be a measurement error until the measurement
was rebuilt. None was spotted by eye — each was caught by a control set up in advance to
catch it. That track record is the most transferable thing in the project.

| what went wrong | found by | what it had faked |
|---|---|---|
| the answer was in the training data | 01/try08 | every result before that point |
| trained ~3x too long with nothing to check against | 01/try09 | the signature of memorization |
| scoring returned the most common answer, not an estimate | 01/try10 | every enrichment number so far |
| scores depended on which wrong answers were in the list | 01/try15 | all comparisons between attempts |
| the same disease appeared twice under a different name | 03/try18 | Fabry's top-0.2% result |
| ranking by disease *category* mistaken for by disease | 03/try17, 03/try25 | Fabry *and* Gaucher |
| guessing which data was held out instead of checking | 03/try26 | would have mis-scored 14% of cases |
| "hard" decoys that were actually salts of the right answer | 01/try14 | 3 of 4 new targets' verdicts |

## Still missing

- **Ceiling tests are queued, not finished** — we know where each approach lands, not how
  well it could do unblinded.
- **Gaucher cannot be placed on the shared list.** Its drug, ambroxol, is not registered
  as fully approved in the chemical database the list is built from. Fabry + Gaucher was
  the thesis's stated proof-of-concept pair, so this needs a decision: drop Gaucher, or
  rebuild the list with a hand-curated exception and report it as such.
- **The side-by-side comparison figure is deliberately kept off this repository.**

## Decisions already made

- **Do not add chemistry to TxGNN.** The point is that it uses *only* the disease map.
- **Do not keep building Fabry-specific fixes.** That teaches a narrow trick for one
  enzyme family with no evidence it carries over. A later test showed the failure being
  chased is a property of the base model itself, present with no training data involved —
  so there was never a fix on the data side.

## How this repository is organized

Grouped by project, then by attempt. Each attempt has a dated folder
(`tryNN_DD-MM-YYYY_description`) with the code as it stood, the real cluster logs, and a
write-up of what was tried and why it failed.

| folder | what it is |
|---|---|
| [01_txgemma_predict](01_txgemma_predict/README.md) | the chemistry model (17 attempts) |
| [02_txgemma_chat](02_txgemma_chat/README.md) | the reasoning model (12 attempts, descoped) |
| [03_txgnn](03_txgnn/README.md) | the disease-map model (26 attempts) |
| [04_setup](04_setup/README.md) | one-time installation |
| [05_multi_agent](05_multi_agent/README.md) | source of the trivial chemical baseline |
| [06_equalized_pool](06_equalized_pool/try01_23-11-2026_build-pool/README.md) | builds and freezes the shared 1,933-drug list |
| [07_structure_pipeline](07_structure_pipeline/README.md) | the structural half (not started) |

Model weights, the disease database and cached downloads stay on the cluster and are not
stored here. Code, logs and write-ups only.

The shared drug list is frozen and fingerprinted (checksum beginning `9d89c3ce`), and
every scoring script verifies it before running. Comparing numbers measured against
different drug lists was this project's single largest source of fake results.
