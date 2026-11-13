# Try 10 — test-evaluation edge-type KeyError fix

**Date:** 26-09-2026 (updated 29-09-2026)
**Scripts:** `txgnn_fabry.py` + `txgnn_fabry.sb`
**Jobs:** resubmitted as 22411998/22411999's successors after try09's jobs both FAILED.

**Provenance correction (29-09-2026):** the first resubmission (22454275/22454276,
"the fix" section below) was intended to run this folder's own script, but
`txgnn_fabry.sb` had been copied straight from try09 without updating its internal
python path — it actually invoked try09's original file the whole time, same class of
path-drift bug documented back in try06. This didn't invalidate that run's result for
the `evaluate_fb` fix specifically, since that fix lives in the installed library
(shared regardless of which project script calls it), and both copies of
`txgnn_fabry.py` were identical at that point anyway. It matters for the *second* bug
found in that run (`rank_of()`'s KeyError, see below) — that fix was made only in this
folder's local copy, so the `.sb` path had to be corrected before resubmitting again,
or it would have silently re-run the unfixed try09 file and hit the same error.

## What happened in try09's actual run (jobs 22411998 fabry, 22411999 gaucher)
Both jobs ran training to completion and crashed identically at the final **test**
evaluation step:
```
KeyError: ('drug', 'contraindication', 'disease')
  File ".../txgnn/TxGNN.py", line 313, in finetune
    ... evaluate_fb(self.best_model, self.g_test_pos, self.g_test_neg, ...)
  File ".../txgnn/utils.py", line 619, in evaluate_fb
    neg_score = torch.cat([pred_score_neg[i] for i in dd_etypes])
KeyError: ('drug', 'contraindication', 'disease')
```

This is the same underlying issue documented back in try05 (a disease's held-out test
split can genuinely have zero real edges of a given `dd_etype`, e.g.
`contraindication`) — but a *different* code path than the one try05 patched.
try05 fixed `etypes_train` filtering used during pretrain/training. This crash is in
`evaluate_fb()`, called at the very end of `finetune()` for the final **test** metric —
a spot that was never hit before because try09's own bug (skipping straight to
`finetune()`/full-graph training, no `pretrain()`) is what let training get far enough
to reach this line for the first time. Both Fabry and Gaucher hit it identically,
meaning it's not disease-specific: whichever disease's held-out test split happens to
lack a `dd_etype` will hit this, and evidently both do.

Notably, `evaluate_fb()`'s own per-relation AUROC loop (a few lines above the crash)
already wraps each relation in `try/except: pass` specifically to tolerate a missing
edge type for that one metric — the aggregate `pos_score`/`neg_score` concatenation
just below it was never given the same tolerance. That inconsistency is what makes
this read as a genuine upstream library gap, not something specific to this project's
usage.

## The fix
Patched the installed library directly (not part of this git repo — lives in
`/work/hdd/bisa/sshah21/envs/txgnn2/lib/python3.8/site-packages/txgnn/utils.py`,
`evaluate_fb()`, lines ~614-619): filter `dd_etypes` down to only the edge types
actually present in both `pred_score_pos` and `pred_score_neg` before the
`torch.cat(...)`, mirroring the tolerance the AUROC loop already has:
```python
present_etypes = [i for i in dd_etypes if i in pred_score_pos and i in pred_score_neg]
pos_score = torch.cat([pred_score_pos[i] for i in present_etypes])
neg_score = torch.cat([pred_score_neg[i] for i in present_etypes])
```

## Status
Resubmitted both jobs under the patched environment: fabry = job **22454275**,
gaucher = job **22454276**. **FAILED again after 7:35h/7:32h** — but this time the
`evaluate_fb` fix worked (training AND test-evaluation both completed for all 3 seeds
each). The crash moved further downstream, into this project's own `rank_of()`:
```
KeyError: '543.0'
  File "txgnn_fabry.py", line 97, in rank_of
    preds = out['prediction'][str(disease_idx)]
```
`rank_of()` assumed `eval_disease_centric()`'s returned `out['prediction']` dict is
keyed by `disease_idx` (TxGNN's internal per-type graph index, e.g. `543.0` for
Fabry) — wrong. Loaded the already-saved `eval.pkl` for each seed (training doesn't
need to be repeated, the checkpoints and full prediction dicts were already written to
disk before the crash) and inspected the actual keys directly rather than guessing:
Fabry's key is `'10526.0'` (PrimeKG's raw disease id field, not the internal idx), and
Gaucher's is `'9266_9265_9267_18150'` — a merged multi-id string, since PrimeKG folds
some diseases together from multiple source ontologies. Neither is predictable from
`disease_idx`, and the two diseases don't even share a key *format*.

**Fix**: since `eval_disease_centric(disease_idxs=[idx])` was called with exactly one
disease, `out['prediction']` is guaranteed to have exactly one key regardless of its
format — verified this holds across all 6 saved seed x disease pickles before relying
on it. Changed `rank_of()` to take `next(iter(...))`/tuple-unpack the sole value
instead of looking up by a predicted key string.

Since checkpoints already exist on disk for all 3 seeds x 2 diseases, a resubmission
under this fix will skip retraining entirely (the script already checks for
`ckpt/model.pt` and loads it) and only redo the fast eval+aggregate step.

## Result, computed directly from the already-saved eval.pkl files (29-09-2026)
Before waiting on a resubmission, loaded the 6 already-saved `eval.pkl` files (one per
seed x disease, written by the FAILED run before it crashed in `rank_of()`) and ran the
fixed aggregation logic directly against them -- no retraining needed, real completed
results:

**Fabry**: migalastat ranks a mean of **17.7 / 7,957** drugs (seed 1: 12th, seed 2:
22nd, seed 3: 19th). Strong, consistent, top-0.2% result across all 3 independently
seeded models. This is a real finding, not a bug -- and a large improvement over
try03's early (differently-configured, `attention=False`) result of 2,863rd.

**Gaucher**: much weaker and more mixed, across the 7 real historical/approved Gaucher
drugs tracked:
| drug | mean rank / 7,957 | note |
|---|---|---|
| miglustat | 1251.3 | huge seed variance: 3724 / **7** / 23 -- seeds 2-3 look excellent, seed 1 doesn't |
| eliglustat | 1470.3 | consistent, moderate (top ~19%) |
| velaglucerase alfa | 3771.7 | mid-pack, not distinguished |
| voglibose | 7050.0 | near-bottom (this isn't actually an approved Gaucher drug -- plausibly a stray inclusion) |
| taliglucerase alfa | 5588.3 | poor |
| alglucerase | 5587.3 | poor |
| imiglucerase | 5586.3 | poor |

**Caveat worth resolving before trusting the enzyme-replacement-therapy numbers**:
alglucerase, imiglucerase, and taliglucerase alfa each score **exactly 0.0**,
identically, across all 3 independently-seeded models. Real trained-model outputs
essentially never land on an exact identical float across 3 different random seeds by
chance -- this pattern looks like a missing-embedding or not-in-candidate-pool
artifact (e.g. these are large protein biologics; possible their drug node lacks
features TxGNN's decoder expects) rather than a genuine learned "this drug doesn't fit"
signal. Not yet root-caused. Treat the poor ranks for these 3 specifically as
unverified until checked, independent of the rest of the Gaucher result.

## Status
Path-drift fixed, resubmitted for an official clean log (should be fast -- checkpoints
cached, only eval+report reruns): fabry = job **22544783**, gaucher = job **22544784**.

## Verification: is this a real fix, or papering over missing rare-disease data?
Checked directly against PrimeKG before trusting the patch. Findings:
- PrimeKG has 30,675 `contraindication` edges globally; Fabry and Gaucher both have
  some in the full graph (visible in their `disease_eval` train/valid splits).
- But `disease_eval`'s own split logic puts **only `indication`/`rev_indication`
  edges into each disease's test.csv** — `contraindication`/`off-label use` land in
  train/valid instead, since only the relation being zero-shot-predicted (indication)
  is held out for a single-disease split. Confirmed via
  `cut -d',' -f3 disease_eval_[543.0]/test.csv | sort -u` (Fabry) and the [16430.0]
  equivalent (Gaucher) — both show only `indication`/`rev_indication`.

So `evaluate_fb()`'s crash is a hardcoded assumption (all 3 `dd_etypes` always present
in the test set) that only holds for the `complex_disease` split style, not
`disease_eval` — the split type that's actually correct for this thesis's "predict
this one rare disease's indication" question. The graph data itself is fine; this
isn't a rare-disease data gap. The fix correctly scores against exactly the relation
type `disease_eval` legitimately produces for its test set, rather than working around
a real absence of data.
