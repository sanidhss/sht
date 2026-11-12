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

## Result (partial — 2 of 4 targets done, 12-11-2026)

Job A completed 10-11 01:03. `tyrosinemia` finished 11-11 19:04 (2h05), `cel` finished
12-11 01:31 (7h55). `mastocytosis` and `nsiad` are still queued behind GPU priority;
job C has not run, so there is no `results.csv` yet. Numbers below are read directly
from each target's `step_stopping_summary.json`.

All four targets routed to **stepstop**; the crossval path went unused this try.

### Enrichment percentile vs 500 decoys (lower = better; 50 = chance)

| target | readout | seed 0 | seed 1 | seed 2 | mean | baseline kNN | baseline ridge | cleared baseline? |
|---|---|---|---|---|---|---|---|---|
| cel | digit_exp | 21.6 | 22.4 | 17.8 | **20.6** | **5.8** | 16.6 | **no** |
| cel | greedy | 59.2 | 11.6 | 17.0 | 29.3 | — | — | no |
| tyrosinemia | digit_exp | 8.8 | 39.6 | 57.4 | **35.3** | **26.8** | 45.2 | **no** |
| tyrosinemia | greedy | 36.8 | 32.2 | 43.0 | 37.3 | — | — | no |

**Both finished targets lose to the non-LLM baseline.** That is the bar the user set,
and the fine-tune did not clear it on either. The Tanimoto k-NN baseline — five nearest
training compounds weighted by fingerprint similarity, no neural network — puts imatinib
at the 5.8th percentile of the decoy field against the fine-tuned model's 20.6.

`beats_hard` is 0/3 seeds on every target and every readout. See the next section for
why that number should not be reported as a result.

### The hard-negative test is broken on 3 of the 4 new targets

This was found by reading the eval sets directly rather than trusting the flag, and it
is the most important thing in this try. Two distinct defects, both in hard-negative
*selection*, which picks the top-3 Tanimoto neighbours of the reference drug out of the
pool with no identity or potency check.

**Defect 1 — the "hard negative" is the reference drug.** Tanimoto similarity to a
salt, a tautomer, or a stereo-unannotated copy of a molecule is ~1.0, so the picker
selects the reference drug itself written a different way:

| target | hard negative | what it actually is |
|---|---|---|
| cel | `CS(=O)(=O)O.Cc1ccc(NC(=O)...` CHEMBL1642 "imatinib mesylate" | imatinib + a methanesulfonate counterion — the same drug, 6886 nM vs imatinib's 84.6 nM, an 81x gap that is assay variation, not chemistry |
| tyrosinemia | `O=C1CCCC(O)=C1C(=O)...` and `O=C1CCCC(=O)C1=C(O)...` | the two enol tautomers of nitisinone — nitisinone is a triketone that exists as the enol; 85 nM vs 37 nM |
| mastocytosis | `CN[C@@H]1C[C@H]2O[C@@](C)...` and `CNC1CC2OC(C)(C1OC)...` | staurosporine with and without stereo annotation — one compound, both carrying the identical 14.7 nM label |

So 2 of tyrosinemia's 3 hard negatives are nitisinone, and 1 of cel's 3 is imatinib. The
test asks the model to rank a molecule above itself, with the "right answer" set by which
assay happened to be recorded.

**Defect 2 — the hard negative is genuinely more potent than the drug.** The test assumes
the reference drug is the most potent member of its structural neighbourhood. For
mastocytosis it is not:

```
mastocytosis  ref midostaurin 57.2 nM  ->  hard negatives 14.7, 14.7, 0.2 nM
```

All three are more potent than midostaurin. Midostaurin is the approved drug because of
selectivity and ADME, not raw KIT potency — staurosporine is a more potent pan-kinase
inhibitor and is undruggable for exactly that reason. **A correct model must fail
mastocytosis's `beats_hard`.** nsiad has a milder version: its third hard negative is
3.0 nM against tolvaptan's 3.4 nM, a tie inside assay noise.

Checked against the four original targets for comparison:

```
gaucher   ref 10979 nM  hard  50000 / 10000 / 100000   (one near-tie)
cf        ref  1293 nM  hard   5385 /  1294 /   3990   (one exact tie)
fabry     ref  40.4 nM  hard 390000 / 45000 / 920000   (clean)
attr      ref   839 nM  hard  13000 / 10000 /  12000   (clean)
```

The original four were never this badly posed — the flaw existed but did not bite, because
their analogs happened to be weaker. It is not a try14 regression; try14 is just the first
set of targets that exposed it. **Any published `beats_hard` number for cel, tyrosinemia
or mastocytosis is meaningless.** The enrichment percentiles are unaffected — they are
measured against the 500-decoy field, not against these three molecules.

Fix for the next try, not applied here (the remaining two array tasks are already queued
against the eval sets job A wrote, and changing them now would make the four targets
non-comparable): in `prep_orphan_target_data.py`, reject a hard-negative candidate when
(a) its largest covalent fragment, stripped of stereochemistry, canonicalises to the same
string as the reference's, or (b) its measured potency is not at least ~3x weaker than the
reference's. Where (b) leaves a target with no valid hard negatives — mastocytosis almost
certainly — that target has no hard-negative test, and the honest thing is to report the
enrichment percentile alone and say so.

### Secondary observations

**The greedy readout collapses; digit_exp does not.** On cel, greedy produced 54–57
distinct values across 500 decoys (~89% collapse) against digit_exp's 239–253. Seed 1's
greedy decode gave the identical value 443 to imatinib and all three hard negatives.
Greedy's cross-seed spread on cel is 47.6 against digit_exp's 4.6. Running both readouts
off one forward pass was worth it: on this evidence digit_exp is the one to report, and
greedy's 59.2 on cel seed 0 would have been a misleading headline number.

**The selection audit passed 6/6 seeds** (all four checkpoints re-scored on full
all-variants validation agreed with the step-selected one). That is a clean contrast with
CF in try11, which failed 2 of 3 — so CF's instability is CF's, not the method's.

**tyrosinemia's step selection is unstable even though the audit passed.** Seed 0 stopped
at step 22 with val loss 2.07; seeds 1 and 2 ran to step 168 and reached 0.59/0.64. Seed 0
is on a different curve entirely — its val loss rises monotonically after step 22 while the
others fall throughout. With 46/35/39 distinct validation molecules this is the small-n
regime the 6-fold CV path exists for; 179 compounds cleared the 150 threshold but the
threshold looks too low. The digit_exp spread across seeds, 8.8 to 57.4, follows from it.

## Still outstanding

- `mastocytosis` and `nsiad` queued; job C will collate when they finish.
- Rewrite the Result header to "final" and add the two missing rows once job C runs.
