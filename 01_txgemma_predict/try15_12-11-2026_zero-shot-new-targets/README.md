# try15 — zero-shot base model on all eight targets, one yardstick

**Job 23833457**, submitted 12-11-2026. GPU, 6h reserved, ~2h30m expected.

## Why this exists

try13 established the project's central finding: on the original four targets, apparent
success tracked the **base model's prior**, not the training data. It ran zero-shot —
the unmodified `txgemma-9b-predict` with no LoRA, no gradients — and found that ambroxol
already beat its structural analogs before any training existed.

**try14 added four new targets and never ran that control.** Checked directly:
`grep -l 'zero_shot\|base_model_only\|no_adapter' try14/*.py` returns nothing. So for
`cel` and `tyrosinemia` there are fine-tuned numbers with nothing to compare them
against, and the single most important question about them cannot be answered:

> Is cel's 20.6 enrichment something fine-tuning *achieved*, or something it *degraded*
> from a better base-model number — the Fabry pattern?

Those have opposite implications and the fine-tuned number alone cannot distinguish them.

## The decoy-pool problem, and why this re-runs the original four

`build_decoys()` builds the 500-decoy field from **every other target in
`ALL_TARGETS`**. That list is not a work list — it silently defines the measuring stick.

| run | ALL_TARGETS | decoy pool for a given target |
|---|---|---|
| try13 zero-shot | 4 targets | union of the other **3** |
| try14 fine-tuned | 8 targets | union of the other **7** |

So try13's zero-shot numbers and try14's fine-tuned numbers **are not on the same
scale**. Scoring cel zero-shot against try13's four-target pool would compare it to its
own fine-tuned number using a different ruler — which is precisely the comparison this
run exists to make.

Running all eight here fixes that: every target, old and new, zero-shot and fine-tuned,
lands on one yardstick for the first time. Re-scoring gaucher/cf/fabry/attr is therefore
**not redundant work** — it is what makes the table comparable. Output goes to
`zero_shot_8target/`, leaving try13's `zero_shot_familiarity/` untouched, because the two
are different measurements and overwriting one with the other would destroy the record.

**Any large shift in the original four is itself a finding** — it would mean try13's
conclusions were sensitive to decoy choice, which nobody has tested.

## What it answers

**1. Did fine-tuning help, hurt, or do nothing — per target?** Against the already-measured
fine-tuned numbers (digit_exp, mean of 3 seeds; lower = better, 50 = chance):

| target | zero-shot (try13, 4-pool) | fine-tuned | best baseline | reading |
|---|---|---|---|---|
| gaucher | 18.4 ✓beats analogs | **0.4** | 21.6 | prior, sharpened |
| cf | 79.2 | **28.8** | 29.4 | **real learning, cold start** |
| fabry | 15.2 | 35.2 | 26.4 | **fine-tuning damaged it** |
| attr | 74.6 | 64.2 | 31.0 | nearly nothing |
| cel | **this run** | 20.6 | 5.8 | unknown |
| tyrosinemia | **this run** | 35.3 | 26.8 | unknown |
| mastocytosis | **this run** | pending | 20.8 | unknown |
| nsiad | **this run** | pending | 9.2 | unknown |

**2. Does the no-training screening result generalise past Gaucher?** This is the one
genuinely positive claim the project has, and it is currently **n = 1**. If a second
target shows useful zero-shot enrichment, it becomes a finding. If seven of eight show
nothing, Gaucher is one lucky molecule and should be reported as such.

## What this run cannot tell you

`beats_hard` is **not interpretable** for cel, tyrosinemia or mastocytosis. Their hard
negatives include salt, tautomer and stereo-unannotated copies of the reference drug
itself, and mastocytosis's analogs are genuinely more potent than midostaurin. See
try14's README. The enrichment percentiles are unaffected — they are measured against the
500-decoy field, not against those three molecules.

So for the generalisation question above, read the **enrichment percentile**, and treat
`beats_hard` as meaningful only for gaucher, cf, fabry, attr and nsiad.

## Safety of the run

- No PEFT import, no adapter load, no gradients, no optimizer — the three `adapter`
  matches in the file are all comments. Verified by grep before submitting.
- Writes only to `zero_shot_8target/`. Touches no per-target directory and no adapter.
- `TARGETS` is exported explicitly in the sbatch even though it equals the full list —
  the try05 scope-coupling bug came from relying on an implicit default.
- `train_blind.jsonl` is read **only** to build the decoy pool. Nothing is trained on.
- Verified before submitting: all 8 targets have both `train_blind.jsonl` and
  `eval_set.json`; both files use the same `{prompt, answer}` schema across old and new
  targets; the script parses; the conda python and output directory exist.

## Result

Pending.
