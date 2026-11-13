# Try 01 — Zero-shot BindingDB diagnostic (GLA/GBA1 only)

**Date:** 18-09-2026
**Script:** `score_txgemma.py` (exact version not separately preserved — see try02 for the
next preserved state, which added imatinib/ABL1) + `score_txgemma.sb`
**Jobs:** 22201987 (FAILED, 6s, no log preserved), 22202595 (TIMEOUT, no log preserved),
22204169 (COMPLETED, no log preserved), **22206964 (COMPLETED, log preserved)**

## What it does
Loads `google/txgemma-9b-predict`, prompts it with the `BindingDB_ki` template from
`tdc_prompts.json` for each target (GLA, GBA1), scoring a fixed candidate list
(migalastat, isofagomine, aspirin, caffeine, metformin, ibuprofen) via zero-shot greedy
decoding. Lower predicted score (0-1000 scale) = stronger predicted binder.

## What happened / why it "failed"
Not a crash — a real, negative scientific result. TxGemma ranked **aspirin above
migalastat** for GLA, and generally failed to identify the known correct chaperone.
Root cause (fully diagnosed later): BindingDB — the database TxGemma's `BindingDB_ki`
task was fine-tuned on — has **zero** affinity records of any kind for GLA and only 17
for GBA1. The model was never given any real training signal for these targets, so its
zero-shot output is unconstrained guessing, not a broken pipeline. Verified via
BindingDB's own REST API (`bindingdb.org/rest/getLigandsByUniprot`).

The three earlier job attempts (22201987/22202595/22204169) predate this conversation's
visibility and their logs are not present in the current results directories — likely
setup/debugging runs before the diagnostic script reached a stable state.
