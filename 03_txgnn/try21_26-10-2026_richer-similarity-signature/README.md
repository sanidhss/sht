# try21 — option 3: richer prototype similarity signature

## What
Identical to try16 (same 120 held-out diseases, same cached split, same seeds and
config) except `sim_measure='all_nodes_profile_more'`. That signature adds
phenotypes + exposures to the disease-similarity profile, matching the TxGNN paper's
4-node-type description. A paired comparison against try16: the only variable is the
similarity signature. Results in `.../connectivity_sweep_profile_more/`.

Script `txgnn_connectivity_profile_more.py` = try16's script with 3 changed lines
(diff-checked). Sample is read from try16's folder. The `.sb` invokes this folder's
script (checked). Verified that `disease_phenotype_positive` and
`rev_exposure_disease` exist as disease-sourced edge types in the split (the richer
signature reads them).

## Ground rule
Variants are compared on the **120 sweep diseases only**. The thesis diseases
(Fabry/Gaucher/CF/TTR) are never used to choose a setting (cf. try07's attention=True
being chosen after seeing Fabry). After a winner is chosen on the sweep, it's applied
once to the thesis diseases.

## Status
- Job **23231153** submitted 26-10-2026 (24h). Analyse with try16's `analyze.py`
  (point it at this results dir) and try19's `rerank.py` (handles this sweep already).
- "2 and 3 together" is covered on the reliability side by try20, which reports the
  flag under both signatures. On the ranking side, try19's correction runs on this
  sweep too.
