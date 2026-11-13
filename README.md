# Overview

Orphan diseases affect fewer than 1 in 2,000 people, so they rarely attract the funding a
drug-development programme needs. Many of them are *conformational* diseases: a mutation
doesn't delete an enzyme, it destabilises it, so the protein misfolds and gets degraded
before it can do its job. That creates an unusual therapeutic opening. A small molecule
that binds the unstable protein and holds it in the right shape — a **pharmacological
chaperone** — can rescue the enzyme the patient already has. Migalastat does this for
α-galactosidase A in Fabry disease; tafamidis does it for transthyretin.

Finding such molecules is slow, and several recent AI systems claim to shortcut it by
predicting which existing drugs might treat which diseases. The catch is that those
predictions are made at a level of abstraction with no physical chemistry in it. A model
can assert that drug X is a candidate for disease Y without any account of whether X fits
the binding site, in a plausible pose, with favourable energetics.

**This thesis asks two questions.** Can these models nominate plausible stabilising
molecules for orphan disease targets? And when they do, does the prediction survive being
checked against structure and energetics?

## The approach

Three AI systems are run independently, chosen because each reasons from a completely
different kind of evidence:

| model | referred to as | what it reasons from | what it never sees |
|---|---|---|---|
| **TxGNN** | the *graph* arm | a network of known disease–gene–drug relationships | any chemical structure |
| **TxGemma-predict** | the *chemistry* arm | molecular structures and measured binding data | the relationship network |
| **TxGemma-chat** | the *reasoning* arm | mechanism descriptions written in plain language | either of the above |

**A naming warning, since it causes real confusion:** two of the three are called TxGemma,
but they are not variants of one experiment. **TxGemma-predict** is a 9-billion-parameter
model fine-tuned to output a *number* — how strongly a given molecule binds a given
protein. **TxGemma-chat** is a separate 27-billion-parameter model that answers in *prose*,
and is asked to state what physical property a disease mechanism requires and then screen
candidates against that stated rule. Different sizes, different tasks, different outputs,
different failure modes. **TxGNN** is unrelated to both: a graph neural network over a
biomedical knowledge base, with no access to chemistry whatsoever.

Below, each is named explicitly rather than by role, so no claim is ambiguous about which
model it describes.

Because the inputs don't overlap, the arms serve as independent checks on each other. Each
is held blind to the known answer for whichever target is being tested — the true
drug–disease link is deleted before training, and the model has to recover it.

The work is then in two parts. **Part 1** runs the models and establishes whether their
rankings mean anything. **Part 2** takes the surviving candidates and characterises them
physically: predicted structures of each candidate bound to its target, interface
energetics, and a map of the specific contacts it makes — compared against the contacts a
validated chaperone makes in a real crystal structure. Part 2 is the point of the thesis;
Part 1 exists to decide which candidates are worth that effort, and to establish whether
the models were run honestly.

## What has been found

The short version: **every apparent success turned out to come from something other than
the ability being claimed — and in each case the real explanation was identifiable.**

- **TxGNN** (graph) appeared to rank Fabry's drug in the top 0.2% of 8,000 candidates. It
  was reading the answer off a **mislabelled database entry**: a different, rare disease
  had been annotated with all of Fabry's genes *and* its drug, so deleting Fabry's own
  link left the answer sitting next door. Removing that entry dropped the drug from 16th
  to 3,170th.

- **TxGemma-predict** (chemistry) genuinely succeeded on Gaucher disease. But running it
  with **no training at all** showed it already favoured the right molecule beforehand —
  the knowledge came from the model's general pre-training, not from the binding data it
  was given. Fine-tuning sharpened an existing prior rather than teaching it anything.

- **TxGemma-chat** (reasoning) depended on **how the question was phrased**. Holding the
  disease, the model and the candidate library fixed, and changing only whether the
  mechanism was described in physical or pharmacological terms, moved the correct drug
  from 11th to 2,974th.

Alongside that, a recurring and slightly humbling result **for TxGemma-predict**: simple
structural similarity often beats it. Ranking candidates by their resemblance to known
active compounds — no neural network involved — won on four of the six targets measured so
far. TxGemma-predict won clearly on one (Gaucher) and narrowly on another.

**TxGNN has not yet faced an equivalent test**, and this is a gap rather than a win. A
structural-similarity baseline isn't meaningful for it, since it never sees chemical
structures at all; the fair comparison is against simple drug popularity, because
well-connected drugs tend to rank highly everywhere. That test is built but has not
returned, so TxGNN's performance should be read as unbenchmarked.

### Two deeper limitations

Both were measured rather than assumed. Both apply specifically to **TxGNN**, because both
are consequences of learning from a relationship database rather than from chemistry.

**Drug databases record what a drug is *used for*, not what it *binds*.** Pharmacological
chaperones are almost all repurposed or still investigational, so the database contains no
link between them and the disease they chaperone. Diflunisal and tolcapone are documented
transthyretin stabilisers in clinical studies; in the database, diflunisal connects only
to arthritis and tolcapone only to Parkinson's. A model trained on approved indications
therefore cannot be tested on chaperones at all — there is no link to hide and then
recover. This does not affect TxGemma-predict, which learns from measured binding data and
so is indifferent to whether a drug was ever approved for anything.

**Hiding the answer properly can remove the information the model needs.** TxGNN scores a
disease by borrowing from its most similar diseases. But a disease's closest relatives
usually share its treatment, so they have to be hidden too — and once they are, what's left
to borrow from is unrelated. In the most recent run, every blindness check passed and TxGNN
was left comparing leukaemias to congestive heart failure.

---

# Library index

Organised by project, then by try — one folder per distinct attempt, named
`tryNN_DD-MM-YYYY_short-description`. Each contains the scripts as they existed at that
point, the real SLURM job log, and a README explaining what it did and, where it failed,
why. Failures are kept deliberately; several of the findings above came from runs that
were recorded as failures first.

The per-project READMEs carry the detail — numbers, configurations, and the full arc of
what was tried. Start there rather than here if you want specifics.

- **[01_txgemma_predict](01_txgemma_predict/README.md)** — **TxGemma-predict**, the
  chemistry arm. Fine-tunes `txgemma-9b-predict` (9B parameters) to score how strongly a
  molecule binds a target protein, from structure and measured bioactivity. Outputs a
  number. 15 tries, 8 targets.
- **[02_txgemma_chat](02_txgemma_chat/README.md)** — **TxGemma-chat**, the reasoning
  arm. Asks `txgemma-27b-chat` (27B parameters, a different model from the above) to state
  what physical property a mechanism requires, then screen candidates against that stated
  rule. Outputs prose. 11 tries. Descoped from the main comparison and kept as a negative
  result.
- **[03_txgnn](03_txgnn/README.md)** — **TxGNN**, the graph arm. Link prediction over
  PrimeKG, with no access to chemical structure. 24 tries; the longest debugging arc, and
  the source of the mislabelled-entry finding.
- **[04_setup](04_setup/README.md)** — one-time environment and data setup.

Model weights, the PrimeKG database, caches and other large artifacts are **not**
duplicated here; they live under `/work/hdd/bisa/sshah21/`. This library holds code and
logs only.
