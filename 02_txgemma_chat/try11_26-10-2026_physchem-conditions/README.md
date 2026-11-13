# Try 11 — does condition *phrasing register* cause the fabry-only result?

**Date:** 26-10-2026
**Scripts:** `screen_physchem.py` + `screen_physchem.sb` (main arm) +
`screen_fabry_variants.sb` (controls) + `descriptor_baseline.py` (no GPU)
**Jobs:** 23226288 (rewrites, COMPLETED 30-10-2026, 1h12m) and
23226442 (fabry controls, COMPLETED 30-10-2026, 1h30m), both exit 0.

## Question

try10 found the stage-2 screen works on fabry (rank 4/3420) and fails at chance on
gaucher/cf/attr. The diagnosis offered was that fabry's condition is the only one phrased
as a *physicochemical property specification*; the others are pharmacological, generic or
vague. That was consistent with the data but tested on n=1.

Two arms here:
1. **Rewrites** — recast gaucher/cf/attr conditions as physicochemical specs, derived
   from each prompt's own Biochemical context and not from the true drug's structure
   (`physchem_conditions.json`, with per-fold circularity notes).
2. **fabry controls** — perturb the *one fold that works* in two directions, holding
   disease, model and library fixed:
   - **Arm A:** fabry's condition rewritten in gaucher's pharmacological register.
   - **Arm B:** fabry's condition paraphrased within the physicochemical register, with
     the word "sugar" removed so the model cannot key on it.

## Result 1: phrasing register is causal, and the effect is enormous

| fabry condition | rank | enrichment | yes-set |
|---|---|---|---|
| original authored (try10) | 4 / 3420 | 427x | 14 |
| **Arm B — paraphrased physicochemical** | **11 / 3420** | 155x | 30 |
| **Arm A — pharmacological register** | **2974 / 3420** | **0.6x** | 339 |

Same disease, same weights, same library, same prompt template. **Only the register of
the condition changed, and migalastat moved from rank 11 to rank 2974** — a ~270x swing,
and Arm A is *worse than random* (0.6x).

Arm B holding at rank 11 shows the result survives rewording *within* the register, with
real but modest sensitivity (4 -> 11). So the try10 diagnosis is confirmed: the screen
works on physicochemical specifications and fails on pharmacological ones, and this is a
property of the **phrasing**, not of fabry or of migalastat.

## Result 2: physicochemical phrasing is necessary but NOT sufficient

| fold | authored (try10) | physchem rewrite | change |
|---|---|---|---|
| gaucher | 1143 / 3421 (1.5x) | 1219 / 3421 (1.4x) | none |
| cf | 993 / 3420 (1.7x) | 640 / 3420 (2.7x) | modest |
| attr | 1774 / 3420 (1.0x) | **291 / 3420 (5.9x)** | **6x better** |

Rewriting helped attr substantially (52nd percentile -> 8.5th), cf modestly, and gaucher
not at all. None reaches recall@100. So register explains *whether the model engages with
the condition at all*, but engaging is not the same as succeeding.

Two secondary observations:
- **cf's yes-set collapsed 514 -> 25** while its rank improved only modestly. The rewrite
  made the condition far more selective without making it more *correct* — the "no basic
  amine, low TPSA/MW, logP >= 3" clauses did push out the kinase inhibitors that
  dominated try10's cf ranking, but what replaced them still is not lumacaftor.
- **gaucher's yes-set grew 1316 -> 1614 (47.2% of the library).** My rewrite is worse than
  the original as a *specification*, despite being physicochemical. "Small lipophilic
  amphiphile with a basic nitrogen" describes roughly half of all approved drugs. A
  physicochemical spec can still be useless if it is not selective.

## Result 3: the descriptor baseline beats the model on all four folds

`descriptor_baseline.py` (RDKit only, no GPU, ~1 min). Each filter is a hand translation
of that fold's condition into descriptors, written from the condition text alone.
Compared on *compounds you would have to test*:

| fold | TxGemma best | RDKit | RDKit advantage |
|---|---|---|---|
| fabry | 4 | **2 / 69** | 2x |
| gaucher | 1219 | **235 / 531** | 5x |
| cf | 640 | **93 / 250** | 7x |
| attr | 291 | **6 / 88** | 48x |

RDKit wins on every fold, by 2x to 48x. Crucially **attr rules out "the condition is
uninformative"** as the explanation for that fold: descriptors find tafamidis at rank 6
from the same condition text where the model manages 291. The information was present;
the model did not use it.

Caveat, flagged before the run: the attr filter requires a biaryl with an ionizable acid
at 250-400 Da, which is close to the documented TTR stabilizer pharmacophore, so that
fold's margin is partly my prior knowledge. fabry and cf are cleaner and RDKit still wins.
The filters were also written after seeing which folds failed, so the baseline is
advantaged overall — that weakens the *margins*, not the direction.

## What this establishes

1. **Condition phrasing register causally controls whether the model works** — the
   cleanest causal result this track has produced (2974 vs 11 on one fold, one variable).
2. **Physicochemical phrasing is necessary but not sufficient.** Three physicochemical
   rewrites still fail; selectivity matters independently of register.
3. **A deterministic property calculator beats the 27B model on all four folds.** Where
   the model appears to work, it is approximating a property filter, more expensively and
   less accurately.
4. Taken with try09 (hallucinated conditions *steer* the screen) the picture is coherent:
   this model converts property specifications into rankings, does it worse than RDKit,
   and cannot be trusted to author the specification itself.

## What is still untested, specific to chat

Everything so far scores a candidate against a *distilled condition string* plus a SMILES
— which is a templated input the `predict` variant could equally take. The capability
unique to `chat`, free-text context, has never been exercised in the mode that works:

- **Score against the full Biochemical context paragraph**, skipping the condition step
  entirely. That step is both an information bottleneck and the hallucination entry point.
- **Include the protein sequence.** The screening prompt omits the target completely;
  drug-target interaction is a core TDC task family and is untested here in scoring mode.
- Sampling/self-consistency (everything so far is `do_sample=False`, one pass).
- Pairwise comparison rather than absolute yes/no verdicts.

The first two are one job and would be the fair test of `chat` as `chat`. If direct
mechanism scoring with the sequence present still loses to RDKit on three folds, this
track is finished and the negative is clean and well-controlled.
