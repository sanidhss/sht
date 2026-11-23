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

## Result (job 23833457, completed 17-11-2026, 53m15s)

### Zero-shot vs fine-tuned, same 8-target decoy pool

For the four **new** targets both numbers come from the same decoy pool, so the
difference is real. digit_exp percentile, lower = better, 50 = chance.

| target | zero-shot | fine-tuned | delta | what fine-tuning did |
|---|---|---|---|---|
| cel | 94.2 | 20.6 | **+73.6** | huge genuine learning |
| nsiad | 96.6 | 16.0 | **+80.6** | huge genuine learning |
| tyrosinemia | 86.6 | 35.3 | **+51.3** | large genuine learning |
| mastocytosis | 14.6 | 77.9 | **−63.2** | **destroyed a working prior** |

**This overturns the reading that fine-tuning adds nothing.** On three of the four new
targets the base model starts near or below chance — cel at 94.2 and nsiad at 96.6 are
not "no signal", they are *actively wrong*, ranking the real drug among the least potent
of 501 — and LoRA moves them 50 to 80 percentile points. That is the largest effect
measured anywhere in this project. try13's conclusion was drawn from four targets of
which two (gaucher, fabry) had strong priors, and it generalised a special case.

**mastocytosis is a second Fabry.** It is the only new target where the base model
already had a good prior (14.6), and fine-tuning wrecked it — 14.6 to 77.9, the single
worst degradation on record, worse than Fabry's 15.2 to 35.2.

### The pattern that actually holds

Sorting every target by its zero-shot score makes the rule visible:

| target | zero-shot | after fine-tuning | direction |
|---|---|---|---|
| fabry | 10.0 | 35.2 | **worse** |
| gaucher | 11.0 | 0.4 | better |
| mastocytosis | 14.6 | 77.9 | **much worse** |
| cf | 47.8 | 28.8 | better |
| attr | 58.0 | 64.2 | flat |
| tyrosinemia | 86.6 | 35.3 | much better |
| cel | 94.2 | 20.6 | much better |
| nsiad | 96.6 | 16.0 | much better |

Everything that starts badly improves sharply. Two of the three targets that start well
are damaged. Gaucher is the lone exception — it starts well and improves.

Nothing lands below ~16 after fine-tuning except Gaucher, and nothing stays above ~65.
Fine-tuning pulls every target toward a band around 20–35 regardless of where it
started. **It is not learning the target so much as learning the task's output
distribution.** That is a sharper and more defensible claim than "it does nothing", and
it explains the damage cases: a prior better than the band gets dragged down to it.

### Against the non-LLM baseline, which is the bar

| target | fine-tuned | best baseline | winner |
|---|---|---|---|
| gaucher | **0.4** | 21.6 | model |
| cf | 28.8 | 29.4 | tie |
| fabry | 35.2 | 26.4 | baseline |
| attr | 64.2 | 31.0 | baseline |
| cel | 20.6 | 5.8 | baseline |
| mastocytosis | 77.9 | 20.8 | baseline |
| nsiad | 16.0 | 9.2 | baseline (model wins 2/3 seeds: 7.2, 3.4) |
| tyrosinemia | 35.3 | 26.8 | baseline (model wins 1/3 seeds: 8.8) |

**1 win, 1 tie, 6 losses.** The large learning above is real and still not enough —
every target it rescues, a Tanimoto k-NN had already solved better. nsiad is the closest
call: two of three seeds beat the baseline, but seed 0 lands at 37.4 and the spread is
34.0, so the mean loses.

### The zero-shot screening claim is still n = 1

Two targets pass `beats_hard` zero-shot — gaucher and mastocytosis — but **mastocytosis's
pass is the model being wrong.** Its hard negatives are genuinely *more* potent than
midostaurin (14.7 / 14.7 / 0.2 nM vs 57.2), so ranking the drug above them is a factual
error that the flag records as a success. See try14's README.

Gaucher remains the only legitimate zero-shot pass in eight targets. The no-training
screening result does **not** generalise.

### Decoy-pool sensitivity: try13's numbers were not stable

This run predicted that a large shift in the original four would itself be a finding.
There is one.

| target | try13 (4-target pool) | try15 (8-target pool) | shift |
|---|---|---|---|
| gaucher | 18.4 | 11.0 | −7.4 |
| cf | 79.2 | 47.8 | **−31.4** |
| fabry | 15.2 | 10.0 | −5.2 |
| attr | 74.6 | 58.0 | −16.6 |

The same model scoring the same drugs moved by up to 31 percentile points purely because
the decoy field changed. All four moved in the same direction — a wider, more chemically
diverse pool contains more molecules the model scores as potent, which lifts the
reference's rank.

**Consequence: an enrichment percentile is only meaningful relative to a stated decoy
pool, and no number in this project should be compared across tries unless both used the
same `ALL_TARGETS`.** The relative ordering survives (gaucher and fabry good, cf and attr
poor, under both pools), so try13's *conclusions* hold; its *numbers* do not transfer.
Any thesis table must state the pool.

## What this changes

1. **"Fine-tuning does nothing" is wrong and should not be written down.** It produces
   the largest effects in the project. It just converges every target to the same
   mediocre band.
2. **Damage is a real and repeated failure mode**, not a Fabry quirk — 2 of the 3
   targets with good priors were degraded.
3. **The baseline verdict is unchanged**: 1 win, 1 tie, 6 losses.
4. **Zero-shot screening does not generalise.** Gaucher is one molecule, not a method.
