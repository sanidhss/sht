# try22 — option 4: learned prototype-blending gate (2×2 with the similarity signature)

## What
Completes a 2×2 on try16's exact 120-disease split (paired; only the listed settings
change):

| | `agg_measure='rarity'` (fixed 0.9 for held-out) | `agg_measure='learn'` (learned gate) |
|---|---|---|
| `all_nodes_profile` | try16 (job 23210549) | **try22 `learn`** (job 23231350) |
| `all_nodes_profile_more` | try21 (job 23231153) | **try22 `learn_profile_more`** (job 23231351) |

`agg_measure='learn'` replaces the fixed rarity weight with
`sigmoid(W_gate[disease]([own ; prototype_blend]))`, a learned, per-disease decision
on how much to borrow from the 3 prototypes. In principle this lets a held-out disease
lean less on poor prototypes (TTR's cos≈0.03) and more on good ones.

## Library bug found and patched (installed env, NOT in git)
**As shipped, `agg_measure='learn'` does not learn.** In
`/work/hdd/bisa/sshah21/envs/txgnn2/lib/python3.8/site-packages/txgnn/model.py`,
`DistMultPredictor.__init__` (L48–53) builds `self.W_gate` as a plain Python `dict` of
`nn.Linear` layers. PyTorch only registers submodules assigned as attributes or held
in `nn.ModuleDict`, so these layers are:
- **not in `model.parameters()`**, so the optimizer (`TxGNN.py:220`,
  `AdamW(self.model.parameters())`) never updates them. The gate stays at its random
  xavier init.
- **not in `state_dict()`**, so they're not saved, and a reloaded model gets a new random gate.

Confirmed directly: a module holding a Linear in a dict reports no parameters for it
(`named_parameters()` / `state_dict()` list only the attribute-held layer).

**Patch** (inserted after L53; backup at `model.py.bak_pre_try22_learn_gate`):
```python
            if agg_measure == 'learn':
                self.W_gate = nn.ModuleDict(self.W_gate)
```
It's conditional on purpose. For every other `agg_measure` (all runs so far use
`rarity`), the state_dict and RNG draw sequence are byte-identical to before, so all
existing checkpoints still load and try19's reload-reproduction check is unaffected.
The patched file compiles. Both scripts **assert at startup** that `W_gate`
parameters are registered, so a missing patch fails immediately instead of silently
training with a random gate.

Note: `W_gate` is created (consuming RNG) even under `rarity`. That was already the
case before the patch and is unchanged.

## Files
- `txgnn_sweep_learn.py` / `.sb`, `txgnn_sweep_learn_profile_more.py` / `.sb`:
  try16's script with only sim/agg settings, NAME, and the startup assertion changed.
  Sample read from try16's folder. Each `.sb` invokes its own script (checked).
- Analysis: `SWEEP=connectivity_sweep_learn python try16_.../analyze.py` (`SWEEP`
  env var added for all 4 arms); try19 `rerank.py` covers all 4 arms.

## Ground rule
Arms are compared on the 120 sweep diseases only. The thesis diseases are never used
to pick a setting. The winner is applied once to them afterwards.

## Status
Submitted 26-10-2026: **23231350** (`learn`), **23231351** (`learn_profile_more`), 24h each.
The `learn` code path had never been exercised in this env before, so watch the first
log lines for runtime errors (they'd appear within the first epoch).
