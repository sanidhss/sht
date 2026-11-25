# try17 — unblinded, on the frozen equalized pool

**Jobs 24534036** (GPU array 0-5, 16h) **→ 24534037** (CPU collate, afterany).
The CPU arms — Tanimoto blinded/unblinded and random chance — have already run.

Unlike try16 this arm **retrains**. try16's adapters were trained blind and there is no
way to unblind them after the fact, so the unblinded model arm is a new fine-tune with
every try11 hyperparameter held fixed. The reference drug's presence in training is the
only variable.

## What "unblinded" means here

`train_blind.jsonl` has the reference held out by prep. Unblinding injects it back with
its **own measured label** and the same 5x SMILES augmentation every other compound got.
Nothing else about the data changes.

This is a **ceiling measurement, not a result.** It asks: given the answer in its
training data, can the method reproduce it? A method that still cannot rank the drug well
when the drug's own measurement is in training has a representation problem, not a data
problem.

## Two ways this silently fails, both found by measurement and both fixed

Injecting the reference is not sufficient. Two mechanisms would have re-blinded it while
the run still called itself unblinded:

**1. The scaffold holdout deletes it.** try09's holdout removes every compound sharing a
Bemis–Murcko scaffold with a hard negative. The reference's hard negatives are its close
analogs, so on **fabry and cel the reference's own scaffold is in that set** — the holdout
would have removed the compound just injected:

| target | reference hit by the holdout? |
|---|---|
| cf | no |
| **fabry** | **yes** |
| attr | no |
| **cel** | **yes** |
| mastocytosis | no |
| tyrosinemia | no |

The reference is now exempted from the holdout when `UNBLIND=1`. Hard negatives stay held
out, and this try reports no hard-negative result anyway.

**2. The scaffold split puts it in validation.** The validation split is by scaffold
group, so the reference can land in validation rather than training — trained-on-nothing,
for that seed. Measured across the 3 seeds:

| target | seeds where the reference fell into validation |
|---|---|
| cf | 1 of 3 |
| **fabry** | **2 of 3** |
| attr | 0 of 3 |
| cel | 0 of 3 |
| mastocytosis | 0 of 3 |
| tyrosinemia | 1 of 3 |

4 of 18 seed-runs. The reference is now pinned into the training split.

**Without both fixes, "unblinded" would have been false on half the targets.** The script
also exits 6 rather than reporting an unblinded number if the reference is not in the
training pool, so this fails loudly rather than silently.

## CPU results, already in

### Tanimoto 5-NN, blinded vs unblinded, and random chance

Rank among the 1,933 approved drugs. Lower is better; random chance is 50%
(verified empirically over 20,000 trials: mean 49.90%, expected rank 967).

| target | drug | real nM | blinded | unblinded | max sim | |
|---|---|---|---|---|---|---|
| mastocytosis | midostaurin | 57.2 | 0.57% | 0.72% | 0.692 | |
| cel | imatinib | 84.6 | 17.80% | 19.35% | 0.942 | near-trivial |
| tyrosinemia | nitisinone | 37.0 | 18.42% | **8.02%** | 0.633 | |
| attr | tafamidis | 838.7 | 57.32% | **24.93%** | 0.435 | |
| cf | lumacaftor | 1293.0 | 71.03% | 71.08% | 0.575 | |
| fabry | migalastat | 40.4 | 85.83% | 65.39% | **1.000** | **TRIVIAL** |
| gaucher | ambroxol | 10979.1 | not in pool | not in pool | 0.192 | clean |
| nsiad | tolvaptan | 3.4 | not in pool | not in pool | **1.000** | **TRIVIAL** |

### Finding 1 — unblinding does not let a 5-NN ace the task

The expected result was rank ≈ 1: the reference is found at Tanimoto 1.000 and reads its
own label back. It does not happen. On cf, cel and mastocytosis unblinding changes
nothing (71.03→71.08, 17.80→19.35, 0.57→0.72); on fabry it stays bad.

Two reasons, both structural:

- **k = 5 averaging dilutes the exact match.** The self-match carries weight 1.0 against
  four neighbours summing to ~2.0, so the prediction is pulled two-thirds of the way
  toward the neighbourhood mean. Label recovery, blinded → unblinded: gaucher 37→42,
  cf 43→44, cel 51→55, mastocytosis 102→96, attr 251→162, tyrosinemia 151→95,
  fabry 600→506, nsiad 147→115. Only attr, tyrosinemia and nsiad improve materially.
- **The rank is set by the pool's scores, not the reference's.** Even a perfect prediction
  of 584 for lumacaftor ranks 1,373rd if the method predicts 1,372 other approved drugs as
  more potent.

### Finding 2 — Morgan-2 cannot see what distinguishes migalastat or tolvaptan

fabry's five nearest actives, unblinded, with similarities and labels:

```
(1.000, 925)  (1.000, 754)  (1.000, 993)  (1.000, 196 <- migalastat itself)  (0.857, 630)
```

**Four neighbours at similarity exactly 1.000.** Morgan-2 is stereo-blind, so migalastat's
three diastereomers are *identical* to it in this representation while their real labels
are 925, 754 and 993 against migalastat's 196. The unblinded prediction is 702 against a
true 196.

nsiad is the same: two neighbours at 1.000, labels 695 and 291.

This is the definitive explanation of why Fabry was never solvable by a fingerprint
method, and it is a representation failure, not a data failure. No amount of data fixes a
descriptor that cannot distinguish the molecules. It also means the ≥0.95 leak flag is
firing on genuine stereochemical degeneracy rather than on a duplicate record.

### Finding 3 — the rank metric is confounded with absolute potency

Three reference drugs are **weak binders**: ambroxol 10,979 nM, lumacaftor 1,293 nM,
tafamidis 839 nM. In a potency ranking of 1,933 drugs a *correct* prediction places them
mid-pack or low. Their poor ranks are partly the right answer.

This affects every enrichment number in this project, not just try17. Where the reference
is weak, read `abs_label_error`, not the rank. The CSV carries both.

It also reframes the thesis question. "Rank the chaperone above other drugs" presumes the
chaperone is a potent binder. For Gaucher — the project's one success — ambroxol binds at
11 µM, so ranking it top is arguably rewarding a *wrong* prediction.

## Scope

Six targets, not eight. gaucher and nsiad cannot be ranked on this pool because ambroxol
and tolvaptan are absent from it by canonical structure (try16's README has the detail;
nothing was substituted). That limitation is about the pool, not about blinding — their
blinded/unblinded label recovery is reported above and in the CSV.

## No hard-negative result

`beats_hard` is absent from the scripts and the outputs. try14 established the selection
is broken on 3 of 4 new targets — it picks salts, tautomers and stereo-unannotated copies
of the reference, and on mastocytosis negatives genuinely more potent than midostaurin, so
a correct model must fail. Carried forward from try16 for the same reason.

## Verification before submitting

- Pool sha256 re-verified in both scripts; mismatch exits 2.
- Holdout exemption and split pinning tested independently in RDKit, results in the two
  tables above.
- `sys.exit` was used four times with no `import sys` — caught by an import audit before
  submission, which would have crashed every array task at its first guard.
- Import audit clean: no unused imports (`MaxMinPicker` removed with the decoy arm;
  `build_decoys` and `enrichment` deleted rather than left dead).
- Collate dry-run produces a valid CSV and SUMMARY.md with the GPU arm absent.
- **`DRY_RUN=1` run of the submitted script itself**, all six targets, exit 0: reference
  injected with the correct label and 5 lines everywhere, `reference in training? True`
  everywhere, holdout exemption firing on exactly fabry and cel, pool sha256 verified and
  the reference located in the pool on every target. This checks the real code path, not
  the independent RDKit replication used for the two tables above.

### One thing the dry-run surfaced

fabry's scaffold holdout removes **18 of 90 compounds (20%)**, against 0.2-3.4% on every
other target:

| target | compounds | removed by holdout | trainable |
|---|---|---|---|
| cf | 1451 | 26 (1.8%) | 1425 |
| fabry | 90 | **18 (20.0%)** | **72** |
| attr | 494 | 17 (3.4%) | 477 |
| cel | 1502 | 18 (1.2%) | 1484 |
| mastocytosis | 1502 | 3 (0.2%) | 1499 |
| tyrosinemia | 180 | 3 (1.7%) | 177 |

This is inherent to fabry: 90 compounds that largely share migalastat's iminosugar
scaffold, so holding out the hard negatives' scaffold takes a fifth of the data with it.
It is not a try17 regression -- try11 and try12 did the same -- but fabry trains on 72
compounds and its unblinded result will be the noisiest of the six. Read it with the
seed spread, not as a point estimate.

## Result

Pending — GPU array queued. Fill in from
`/work/hdd/bisa/sshah21/results/txgemma_lora_orphan/try17_unblinded/SUMMARY.md`.

Blinded model numbers already available from try16: cf 38.89%, fabry 33.79% (the fabry
figure averages all six adapters, three step-stopping and three cross-validation; the CSV
separates them by `path_used`).
