# Try 10 — test-evaluation edge-type KeyError fix

**Date:** 26-09-2026
**Scripts:** `txgnn_fabry.py` + `txgnn_fabry.sb` (unchanged from try09 — the actual fix
is a library patch, not a change to this project's own code, see below)
**Jobs:** resubmitted as 22411998/22411999's successors after try09's jobs both FAILED.

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
gaucher = job **22454276**. Result pending.

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
