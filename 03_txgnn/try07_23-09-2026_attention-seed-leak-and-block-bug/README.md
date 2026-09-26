# Try 07 — attention=True + multi-seed: two new, distinct problems found

**Date:** 23-09-2026
**Script:** `txgnn_fabry.py` at the point `attention=True` and `SEEDS=[1,2,3]` were
added — Fabry still on `complex_disease`, Gaucher on `disease_eval` — this exact version
superseded by try09, not separately preserved.
**Jobs:** 22318515 (FAILED, Fabry), 22318516 (FAILED, Gaucher)

## Why attention=True was added
Diagnosed that migalastat's real biochemical signal (a direct `drug_protein` edge to
GLA) was getting diluted: migalastat has 775 total edges in PrimeKG, 770 of them
`drug_drug` interactions — the one edge that matters is a 0.1% minority signal, and the
default `HeteroRGCNLayer` averages all neighbors uniformly with no learned weighting.
`attention=True` swaps in `AttHeteroRGCNLayer`, meant to let the model learn to
prioritize a high-value edge instead of drowning it out.

## What happened / why both failed
**Fabry (22318515)**: `complex_disease` with **seed=1** leaked migalastat's edge into
train (unlike seed=42, which happened to hold it out — see try05). This is itself an
important finding: `complex_disease`'s blindness for a given disease is seed-dependent
luck, not structural. A secondary bug in the script's own `check_blind()` then crashed
trying to print columns (`x_name`/`y_name`) that don't exist on this dataframe — never
exercised before since seed=42 never leaked. (Fixed: switched both diseases to
`disease_eval`, which guarantees blindness by construction regardless of seed; fixed the
column-printing bug.)

**Gaucher (22318516)**: first appearance of a new, real TxGNN library bug —
`KeyError: 'Wh_anatomy_anatomy'` inside `AttHeteroRGCNLayer.edge_attention()`, during
`pretrain()`. Root cause (fully diagnosed by try08/try09): DGL's minibatch "block"
sampling (used by `pretrain()`) stores source and destination nodes of a same-type
relation (e.g. anatomy-to-anatomy) as separate stores even though both sides are the
same node type, and the attention layer's code only ever writes its computed feature to
one side. `finetune()` doesn't hit this because it runs on the full graph, not blocks.
