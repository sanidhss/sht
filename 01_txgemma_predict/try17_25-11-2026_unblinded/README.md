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

## Scheduling: re-submitted as 18 preempt chunks (25-11-2026)

The original submission (`24534036`, one array of 6 tasks x 3 seeds, 16h, `gpuA100x4`)
was cancelled and replaced:

```
24536392  try17_large   9 tasks  cf, cel, mastocytosis x 3 seeds   16h
24536393  try17_small   9 tasks  fabry, attr, tyrosinemia x 3 seeds 6h
24536394  try17_B_collate                          afterany on both
          -p gpuA40x4-preempt,gpuA100x4-preempt    --requeue
```

One array task per **(target, seed)**. Driven by a hard deadline of 12:00 on 27-11-2026.

### What the scheduler actually rewards here, measured

Start estimates from `sbatch --test-only`, which costs nothing and queues nothing:

| time limit | gpuA100x4 | + A40 + H200 |
|---|---|---|
| 1h | Dec 4 01:10 | Dec 4 02:23 |
| 2h | Dec 4 01:41 | Dec 4 01:41 |
| 6h | Dec 4 01:41 | Dec 4 01:41 |
| 16h | Dec 4 01:41 | Dec 4 01:41 |

**For a zero-age job, the time limit and the partition list change nothing.** Priority
is the binding constraint -- a fresh job sits behind ~2,600 pending jobs and cannot win
a reservation at any size. Splitting for backfill alone buys nothing.

**For an AGED job the opposite is true.** try16 (`24440722`, queued since 23-11, priority
2156) had its limit cut 8h -> 2h in place with `scontrol update`, and its start estimate
moved **26-11 15:48 -> 25-11 19:54, nearly 20 hours earlier**. Backfill engages once a
job has the priority to be considered at all. The lesson is the pairing, not the limit:
*shorten aged jobs in place; never resubmit them.*

### Partition choice, and two traps

| partition | free GPUs | pending | fresh-job estimate |
|---|---|---|---|
| **gpuA40x4-preempt** | 5 | 525 | **Nov 26 13:09** |
| gpuA100x4-preempt | 6 | 736 | Nov 27 03:48 |
| gpuA100x4 | 6 | 2596 | Dec 4 01:41 |
| gpuH200x8 | 9 | 690 | **2027-01-17** |
| full | 21 | **0** | **not permitted** |

- **gpuH200x8 is a trap.** Most free GPUs, best free/pending ratio, newest hardware --
  and nearly two months out. Free-GPU count is the wrong metric; those GPUs are held by
  reservations. Excluded.
- **`full` is a trap.** 21 free GPUs and zero pending because
  `AllowAccounts=bbka-delta-gpu,bbka-delta-cpu` -- another project's partition. Not ours.
- Both preempt partitions are listed so SLURM takes whichever frees first.
  `PreemptMode=REQUEUE`, `GraceTime=300`, `PreemptExemptTime=10min`: a preempted job is
  requeued, not killed, but **restarts from scratch**. That, not backfill, is the real
  reason for fine chunks -- a 4h chunk is far likelier to complete between preemptions
  than a 16h one.

### Time limits are sized for A40, not A100

Per-(target, seed) chunk times measured on A100 were 3h13-3h49 (large) and 1h17-1h29
(small). A40 runs ~696 GB/s against A100's ~2039 GB/s, and single-sequence
autoregressive generation is memory-bandwidth bound, so A40 may be **~3x slower** --
up to ~11h30m for a large chunk. Limits are 16h/6h accordingly. This costs nothing:
the limit was measured not to affect the start estimate on these partitions.

**There is no empirical A40 number for this workload.** The only historical A40 job
(`tx_score`, 41-47 min) was almost entirely model load. The 3x figure is a
bandwidth-ratio argument, not a measurement, and it is the largest uncertainty here.

Verified runnable on A40: `torch 2.14.0+cu130` lists `sm_75, sm_80, sm_86, sm_90,
sm_100, sm_120`, so sm_86 is covered; the trainer and scorer both run **batch size 1**,
so 48 GB is ample.

### Code change the chunking required

Three tasks for one target now run **concurrently**, so the per-target
`step_stopping_summary.json` would have been a write race with the last writer silently
dropping two seeds. Each seed now writes
`step_stopping_summary_seed<N>.json`, and `collate_unblinded.py` merges the shards
(falling back to the single-file name if a 3-seeds-in-one-job run is ever used).

### Deadline risk, recorded honestly

Runway 45.5h; estimated start 26-11 13:09, leaving ~23h of compute. The large chunks fit
only if they start near the estimate and are not preempted late. The **small array is the
safe deliverable**; cf, cel and mastocytosis may not land by 12:00 on 27-11.

Two fallbacks, neither applied:
1. Cut the large targets to 1 seed (3 chunks, not 9). Near-guarantees delivery but loses
   the seed-variance estimate -- which has mattered on every target so far, with spreads
   up to 48 percentile points.
2. Batch the pool scoring. The scorer generates one prompt at a time; batching 16-32
   would cut that phase several-fold. Rejected under this deadline: left-padding affects
   per-position logits, so it risks the `digit_exp` readout and would break strict
   comparability with try16. It is the right fix for the next run.

## Result

Pending — GPU array queued. Fill in from
`/work/hdd/bisa/sshah21/results/txgemma_lora_orphan/try17_unblinded/SUMMARY.md`.

Blinded model numbers already available from try16: cf 38.89%, fabry 33.79% (the fabry
figure averages all six adapters, three step-stopping and three cross-validation; the CSV
separates them by `path_used`).
