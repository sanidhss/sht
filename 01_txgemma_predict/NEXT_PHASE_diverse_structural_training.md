# Next phase: diverse, structurally-aware training (planning doc, not yet built)

**Status: groundwork only, as of 29-09-2026.** This document captures the design
decision and rationale so it isn't lost to chat. Nothing described here has been
implemented -- try06 (CF/TTR through the existing, unmodified pipeline) is the actual
next run; this is what comes after it, pending try06's result.

## Why this exists
Fabry's numeric-predict fine-tune collapses on its hard-negative test: it predicts an
identical score for migalastat and all 3 real diastereomers, unable to resolve
stereochemistry at all (see `try05/README.md`'s Analysis section). The tempting fix --
train specifically on diastereomer-discrimination pairs -- was considered and
rejected in its naive form: if 3 of 4 training targets are lysosomal enzymes where
stereochemistry is the dominant discriminator, the model doesn't learn "identify
what discriminates real candidates for *this* mechanism," it learns "stereochemistry
is usually the answer" -- a narrow heuristic that would misfire on CF (scaffold
rigidity vs. a domain interface) and TTR (hormone-channel geometry), neither of which
is fundamentally about chiral centers the way Fabry's transition-state mimicry is.
There is currently no evidence any Fabry/Gaucher-specific fix would extrapolate to a
disease with a genuinely different receptor profile -- which is the actual thing this
thesis needs to work.

## Design decision: general chemistry + physical structural awareness, across diverse diseases
Train across many mechanistically diverse ligand-receptor pairs (not just the 4
orphan-disease targets -- broader ChEMBL target diversity too), so structural concepts
like diastereomer sensitivity and steric fit are learned as *general, recurring
patterns* the model can apply to an unseen receptor, not memorized quirks of one
target's dataset. Two data layers:
1. **Chemistry** (already built): ChEMBL/BindingDB/PubChem bioactivity + SMILES
   augmentation, as in try05.
2. **Physical/structural awareness** (new): explicit training signal for things like
   steric clash and stereospecific fit, computed directly from SMILES via RDKit 3D
   conformer generation (e.g. shape/volume descriptors, stereocenter-aware
   fingerprints, torsion/contact proxies) -- **not** from an external structural
   database.

## Why not PDBbind (rejected, explicit reasoning)
PDBbind (3D structure + affinity, the obvious candidate for "physical" grounding) was
considered and rejected: it only covers already-crystallized protein-ligand
complexes. A production run on a genuinely novel orphan disease will almost certainly
have data gaps there -- the candidate drugs being screened for a brand-new target have
essentially never been solved in complex with it (that's the whole point of screening
them). Training a dependency on PDBbind coverage would repeat the exact mistake this
phase exists to avoid: building something that works on well-characterized diseases
and has no guarantee of holding up where the real data is thin. RDKit-computable
descriptors don't have this problem -- they're computable for literally any SMILES
string, with zero dependency on whether that exact compound has ever been
experimentally characterized against anything.

## Relationship to the downstream pipeline (important framing for evaluation)
This ranking step isn't the final answer -- it's a screen. The actual downstream flow
is: numeric-predict ranking -> biochemical analysis / filtering -> structural
validation (AF3, PyRosetta, Boltz2, potentially wet lab) on the surviving candidates.
The bar for this step is therefore "does the correct drug survive into the top-ranked
shortlist that gets forwarded to structural validation," not "does it rank #1 above
every real structural analog." Worth re-evaluating try05's existing results
(including Fabry's "failure") against a top-K framing rather than strict pairwise
hard-negative discrimination -- separately from building this next phase, since it's
a re-interpretation of data already in hand, not new data.

## Scope, deliberately deferred
- Multi-disease training set composition (which targets beyond the 4 orphan diseases,
  how many, how to avoid the same narrow-bias failure mode at a larger scale).
- RDKit 3D descriptor pipeline: which specific descriptors, conformer generation
  method/cost, how they get folded into the training prompt/objective.
- Whether/how to combine with the try05-style hard-negative test, or replace it with
  a top-K-aware metric per the framing above.
- TxGemma/transformer-only, matching the existing split with TxGNN (which stays pure
  graph-topology, not touched by this).

None of this is scheduled yet -- pending try06's CF/TTR result, which is the actual
test of whether the *current* general approach already extrapolates reasonably before
investing in this heavier build.
