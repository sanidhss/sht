# Try 10 — multi-disease validation of the stage-2 screen

**Date:** 22-10-2026
**Scripts:** `screen_all_targets.py` + `screen_all_targets.sb` (copied from try09, then patched)
**Job:** 23114793, COMPLETED 24-10-2026 04:36 (1h28m, exit 0).
Log: `/work/hdd/bisa/sshah21/results/reasoning_screen/screen_all_23114793.out`

## Why

try08 screened **only fabry** and reported migalastat at rank 3/3420 with 570x
enrichment. That single number has been carrying the claim "stage 2 works" for several
tries. This runs the identical protocol — authored condition handed in, base model, 3420
approved drugs — on all four folds to check whether it generalizes.

## Result: it does not generalize. fabry was a one-off.

| target | true drug | rank | pool | percentile | enrichment | yes-set |
|---|---|---|---|---|---|---|
| **fabry** | migalastat | **4** | 3420 | **0.12%** | **427x** | 14 |
| gaucher | ambroxol | 1143 | 3421 | 33.4% | 1.5x | 1316 |
| cf | lumacaftor | 993 | 3420 | 29.0% | 1.7x | 514 |
| attr | tafamidis | 1774 | 3420 | 51.9% | **1.0x** | 132 |

Random is 1.0x. Three of four folds are at or indistinguishable from chance; attr is
*exactly* chance. recall@10/@50/@100 all miss on gaucher, cf and attr.

**The "stage 2 works" claim recorded in try08 and the project README is falsified as a
general statement.** It holds for one fold out of four.

(fabry reproduced at rank 4 rather than try08's rank 3 — batch size differs, 12 vs 16,
and bf16 batch composition perturbs logits slightly. The top-12 block is margin-packed,
so that shuffle is expected and consistent with try08's recorded instability caveat.)

## Diagnosis: it is the condition, not the model

Reading the four authored conditions against their results, the pattern is clean —
the screen works when the condition is a **crisp, positive, physicochemical
specification**, and fails when it is not.

- **fabry (works):** "compact, highly polar, with a ring position capable of carrying
  positive charge." Every clause maps to a computable molecular property.
- **gaucher (fails, yes-set 38.5% of the library):** "must act as a secondary/modulatory
  chaperone contact rather than a clean sugar-transition-state mimic -- modulation and
  modest potency." This is **pharmacological and negative** — it describes what the
  compound is *not* and how potent it is. There is no structural handle, so the model
  says yes to a third of the library and the ranking is meaningless.
- **cf (fails):** "larger, rigid, multi-ring scaffold." The model applied this
  *correctly* — the top 20 is daclatasvir, lenacapavir, pibrentasvir, nilotinib,
  pazopanib, bosutinib, quizartinib, all large rigid multi-ring molecules. The condition
  simply does not distinguish lumacaftor from the several hundred modern kinase
  inhibitors and antivirals that also satisfy it. A correct but non-selective condition.
- **attr (fails, exactly 1.0x):** "small hormone-like ligand at a hydrophobic interfacial
  pocket, with a polar anchor at the entrance." Partly structural, but the three authored
  attr analogs rank 2146-2358, *worse* than the true drug at 1774 — nothing in this fold
  is being ordered meaningfully.

So stage 2 is better described as: **it converts a physicochemical property
specification into a ranking.** When the condition is one of those, it works very well.
When the condition is pharmacological, comparative, or generic, it returns noise — and
in the cf case it returns a confidently correct application of an unselective rule.

This also retroactively explains try10's companion finding (the RDKit baseline, recorded
below): descriptors matched the model closely on fabry precisely because fabry's
condition is the descriptor-shaped one.

## No usable self-check, again

Yes-set size looked like a candidate quality signal — fabry's 14 (0.4%) against
gaucher's 1316 (38.5%). It does not hold up: **attr has the second-smallest yes-set (132)
and the worst rank (1774, exactly chance)**. A tight yes-set is necessary but not
sufficient, so it cannot be used to tell a good run from a bad one without ground truth.

This is the second candidate warning light to fail (the first, "empty yes-set", was
withdrawn in try09). Both failed the same way: visible on one fold, gone on the others.

## Companion result: the RDKit descriptor baseline (fabry only)

Built from the fabry condition's words alone, never referencing migalastat
("compact" -> MW < 300; "highly polar" -> TPSA/MW; "ring position capable of carrying
positive charge" -> `[#7;R]`):

| approach | migalastat rank |
|---|---|
| polarity/size only | 133 / 3420 |
| + MW < 300 | 93 / 1034 |
| + ring nitrogen | **8 / 456** |
| TxGemma (try08) | **3 / 3420** |

Three lines of RDKit land within ~2.7x of the 27B model on the one fold where the model
works. Given the multi-disease result above, the fair reading is that both methods
succeed on fabry for the same reason — its condition is a property filter in prose — and
the model's measured advantage over descriptors rests on a single fold.

## Script changes from try09

- **Reference-drug spiking.** ambroxol is not `max_phase=4` in ChEMBL (European approval,
  never US FDA), so it is absent from the approved library. Without this the gaucher fold
  would have had no true drug to rank and `rank_of` would have returned `None` silently —
  the same failure class as try05's spurious `top_choice: None`. A hard assert now fails
  the run if the true drug is missing from the pool.
- **Multi-target loop in one process** so the 27B loads once rather than four times.
- `--targets` replaces `--target`; a summary table prints across folds.

## What this changes

1. The only defensible claim left is narrow: **given a crisp physicochemical condition,
   the model ranks a library well on 1 of 4 tested mechanisms**, and a trivial descriptor
   filter nearly matches it there.
2. Authoring quality, not model capability, is the dominant variable. Three of these four
   conditions were written by the same author in the same session and three of them do
   not work as screening specifications.
3. Before any further screening work: the gaucher/cf/attr conditions should be rewritten
   as physicochemical specifications and re-run. That is a cheap and direct test of the
   diagnosis above — if rewriting fixes them, the limitation is authoring; if it does not,
   the limitation is the model.
