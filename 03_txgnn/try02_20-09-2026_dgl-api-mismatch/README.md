# Try 02 — DGL upgraded to fix try01, breaks a different way

**Date:** 20-09-2026
**Script:** `txgnn_fabry.py` — same era as try01, not separately preserved.
**Env:** `envs/txgnn` (dgl upgraded in-place after try01, still same conda env)
**Job:** 22229334 (FAILED)

## What happened / why it failed
`AttributeError: module 'dgl.dataloading' has no attribute 'EdgeDataLoader'`, inside
`model.pretrain()`. Upgrading `dgl` to chase try01's error broke a different part of the
code: TxGNN's library was written against an older DGL API (`EdgeDataLoader`, removed in
newer DGL versions in favor of a unified `DataLoader`). This is what motivated building a
completely separate, modern environment (`envs/txgnn2`: dgl 2.4.0+cu118, torch
2.1.0+cu118) instead of continuing to patch the old one — see try03 onward.
