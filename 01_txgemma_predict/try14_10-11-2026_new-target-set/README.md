# try14 — six new orphan targets, chained to run unattended

**Status: submitted 10-11-2026. Jobs listed at the bottom; results section is pending.**

Extends the pipeline from 4 targets to 8 runnable ones, with automatic routing between
try11's step-level stopping and try12's 6-fold cross-validation based on measured pool
size, as a three-job SLURM chain that needs no attention between submission and
`results.csv`.

## Targets

| target | gene | reference drug | ChEMBL id | outcome |
|---|---|---|---|---|
| cel | PDGFRA | imatinib | CHEMBL2007 | runnable |
| mastocytosis | KIT | midostaurin | CHEMBL1936 | runnable |
| nsiad | AVPR2 | tolvaptan | CHEMBL1790 | runnable |
| tyrosinemia | HPD | nitisinone | CHEMBL1861 | runnable, **179 compounds measured** |
| mh | RYR1 | dantrolene | CHEMBL1846 | **untrainable — not run** |
| pku | PAH | sapropterin | CHEMBL3076 | **untrainable — not run** |

Every id was vetted before use; the full transcript, including the nine rejected ids and
why, is in **`VETTING.md`**. Read that file before changing any target config — it records
three places where the obvious approach is wrong.

## The chain

```
jobA_prep.sb      CPU, 3h    fetch + pool sizes + routing.json + descriptor baseline
      |  afterok
jobB_large.sb     GPU, 18h, array 0-2    cel, mastocytosis, nsiad
jobB_small.sb     GPU,  8h, array 0-0    tyrosinemia
      |  afterany (both)
jobC_collate.sb   CPU, 30m   results.csv + SUMMARY.md
```

`run_routed.sh` is the dispatcher both B arrays call. It reads `routing.json` and runs
whichever harness job A chose for that target, so **the array a target sits in sets only
its time ceiling, never its method** — a target that routes unexpectedly still gets the
correct harness. It refuses to run with `TARGET` unset rather than guessing, per the
standing rule from the try05 scope-coupling bug.

Job C depends on `afterany`, deliberately: an unattended chain is most useful when the
collator still produces a report after a task dies, naming what is missing rather than
failing silently.

## Routing rule

| measured distinct compounds | path | why |
|---|---|---|
| < 40 | **not run**, reported untrainable | a scaffold holdout plus any validation split leaves nothing to measure |
| 40–149 | try12 6-fold scaffold CV | the Fabry regime: an 11-molecule checking set made selection arbitrary (spread 38.4); CV fixed it (spread 4.6) |
| ≥ 150 | try11 step-level stopping | a 15% split still leaves ~22+ validation molecules |

Parameters are try11's, unchanged and set explicitly in the job scripts so the logs record
them: `SEEDS=0,1,2  N_EPOCH=2  LR=6.666666666666667e-05  EVALS_PER_EPOCH=7
VAL_EVAL_CAP=400  N_DECOYS=500`. Every adapter is scored **both** greedy and digit-logit
expectation from the same forward pass, so the readout and the training change stay
attributable. try11's selection audit is kept, and on disagreement the step-selected
checkpoint is still the one scored, so the check stays diagnostic rather than becoming
policy.

## Deviations from the plan, all of them forced by measurement

Four. Each is justified at length in `VETTING.md`; summarized here so nothing is buried.

**1. The `assay_organism = Homo sapiens` filter was not implemented as specified, because
it would have deleted every record of every target.** `assay_organism` is `null` on all
1010 cel / 4554 mastocytosis / 982 nsiad / 68 tyrosinemia / 11 mh / 1 pku activity records
— the field lives on ChEMBL's assay endpoint, not the activity endpoint. The plant-HPPD
trap is real but is already avoided structurally: plant HPPD assays sit under different
target ids with plant organisms, and querying by the human accession P32754 never returns
them. Assay descriptions confirm human enzyme ("Inhibition of human recombinant His-tagged
HPPD expressed in *Escherichia coli* BL21(DE3)" — bacterial *host*, human protein, which a
naive organism filter would also have wrongly discarded). A correct assay-level guard
(`require_human_assay`) was built, caches per assay id, and drops only *explicitly*
non-human records; it is enabled on no target because none needs it.

**2. Three of six reference drugs are unfindable by try06's `pref_name` match.** nitisinone,
dantrolene and sapropterin all fail. Two fixes: `reference_drug_chembl_id` is now the
primary matcher (confirmed working — tyrosinemia reports `matched by molecule_chembl_id
(CHEMBL1337)`), and tyrosinemia gets a per-target `activity_types` override adding **Ki**,
because nitisinone's only measured activity against HPD is a Ki of 37 nM and the pipeline
fetched IC50/EC50 only.

**3. mh and pku are not run, for a stronger reason than pool size.** Both reference drugs
have **zero** measured activities of any type against their target, so there is no anchor
to validate a ranking against. pku additionally does not fit the task at all: sapropterin
**is** PAH's natural cofactor, not an inhibitor, so "half-maximal inhibitory potency" is
not a quantity that exists here — that one would stay untrainable even with thousands of
records. RYR1 is also 5,038 aa, 3.4x CFTR.

**4. The descriptor baseline had to be rebuilt rather than reused.**
`02_txgemma_chat/try11/descriptor_baseline.py` cannot run on a new target and its numbers
are not in these units: it reads `reasoning_chains/<fold>.json` (which exists only for the
four chat folds), carries a **hand-written RDKit filter per fold** translated from that
fold's condition *text* (there is no condition text for a new target, and writing a filter
from the reference drug's own structure would be circular — that script's own docstring
forbids it, and its README notes the filters were written after seeing which folds failed),
and reports **rank in the 3,417-drug approved library** for a text-screening task, whereas
this track reports **enrichment percentile against 500 cross-target decoys**. The "2x to
48x better than the model" result is real in its own units but is not comparable to a
percentile here.

`descriptor_baseline_predict.py` is the same idea rebuilt to be target-agnostic and
reported in identical units — fit on exactly the compounds the LLM trains on, under the
same scaffold holdout, scored against the same 500 decoys:

- **knn** — Tanimoto-weighted k-NN (k=5) on Morgan fingerprints. If a 9B fine-tune cannot
  beat nearest-neighbour lookup, it has learned nothing a lookup table does not have.
- **ridge** — least-squares on 14 RDKit descriptors (numpy only; scikit-learn is not
  installed in this environment and is not needed for 14 features).

Measured already, on tyrosinemia: **knn 21.8 percentile, ridge 47.8**. So 21.8 is the
number the fine-tune has to clear there. Neither baseline beats the hard negatives, which
makes sense — they are the same triketone chemotype, so a fingerprint lookup cannot
separate them.

## One conflict found and fixed

**Imatinib is both `cel`'s reference drug and a member of the fixed `EASY_NEGATIVES`
panel** — it would have been scored as the compound to rank highly *and* as an unrelated
negative control in the same eval set. `build_easy_negatives` now filters the panel against
the reference by canonical structure, so a synonym cannot slip through, and prints what it
dropped.

## A runtime cap was added

`MAX_TRAIN_COMPOUNDS=1500`, applied by **MaxMin diversity selection**, not truncation. KIT
has ~4,350 ChEMBL records plus ~6,175 usable PubChem rows; unbounded that is several times
CF's 1,450 compounds, and CF alone took 9h49m. Truncating ChEMBL's pagination order would
bias the training set toward whichever chemotypes paginate first, so the cap keeps the
subset that spans the widest chemical space instead. try07 already established that volume
is not the lever (Gaucher passed at 493 compounds, down from 1,208). The cap is applied
**after** hard negatives are chosen, and any hard negative the picker drops is re-added, so
the eval set cannot move with the cap. Both raw and capped counts are recorded.

## Time limits, and what they are based on

CF is the calibration point: 977 prompt tokens x 6,050 training lines x 3 seeds = 9h49m.
Measured prompt lengths here are PDGFRA 780, KIT 720, AVPR2 415, HPD 433 tokens — all at
or below CFTR's — and the 1,500-compound cap bounds lines at 7,500. So the largest new run
costs about the same as CF, hence ~10h expected under an 18h ceiling. CF timed out twice
before (10h, 12h); this is sized not to repeat that. The crossval path's 8h is dominated by
model loading: it loads the base model 9 times (6 folds + 3 seeds) at ~25 min each, so
~3.75h is spent loading before any training.

## Comparability caveat

The decoy field is built from every *other* target's compounds, so adding four target names
changes the pool for all of them. try14's percentiles are comparable **within** try14 but
not strictly against try11's, which drew from a 4-target pool. Pool size is printed and
recorded per run.

## Expected vs actual routing

The plan expected pku and possibly mh to route to cross-validation. Neither did — they are
not trainable at all. And tyrosinemia, the one target expected to be small, measured **179
distinct compounds** (113 ChEMBL + 14 BindingDB + 68 PubChem, post-dedup) and routed to
**stepstop**. So the crossval path may go unused this try. It is wired and dispatched from
the routing file rather than from an assumption, so it will be used if `cel`,
`mastocytosis` or `nsiad` come back smaller than expected. That these four orphan targets
have more public binding data than Fabry does is itself a finding.

## What was verified before submitting

- Prep ran end-to-end for real on tyrosinemia: Ki fetch, reference match by ChEMBL id,
  hard-negative selection, augmentation (5.00x), routing decision, `eval_set.json`.
- `descriptor_baseline_predict.py` ran for real on tyrosinemia (knn 21.8 / ridge 47.8).
- `run_routed.sh` dispatches `tyrosinemia -> stepstop` correctly and exits 2 on unset
  `TARGET`.
- Both eval harnesses reject an unknown `TARGET` as their first action.
- `build_decoys` skips targets with no `train_blind.jsonl`, so the untrainable targets
  cannot crash a run that is building its decoy pool.
- All five Python files byte-compile; `run_routed.sh` passes `bash -n`.
- Partitions and accounts confirmed against `sinfo` / `sacctmgr`: `cpu` + `bisa-delta-cpu`,
  `gpuA100x4` + `bisa-delta-gpu`.

## Result

Pending. Fill in from `/work/hdd/bisa/sshah21/results/txgemma_lora_orphan/try14_new_targets/SUMMARY.md`
once job C runs, and state plainly for each target whether it cleared the non-LLM baseline
— that is the bar, not 50th percentile.
