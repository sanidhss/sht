# Try 02 — Zero-shot diagnostic extended to ABL1/imatinib

**Date:** 19-09-2026
**Script:** `score_txgemma.py` (this copy matches the run) + `score_txgemma.sb`
**Job:** 22216414 (COMPLETED)

## What it does
Same as try01, extended with a third target (ABL1) and a third candidate (imatinib) —
testing whether TxGemma at least gets a well-studied kinase/kinase-inhibitor pair right,
as a sanity check that the pipeline itself works when data coverage isn't the limiting
factor.

## What happened / why it "failed"
Also failed, and for a more specific reason than try01. BindingDB has 186 records for
ABL1 (much better coverage than GLA/GBA1) — but the model still didn't rank imatinib
best. Root cause: the one real potent Ki measurement for imatinib in BindingDB (Ki=14)
is filed under **"Breakpoint cluster region protein/Tyrosine-protein kinase ABL1"**
(the BCR-ABL fusion oncoprotein), not plain wild-type ABL1 (P00519, what we queried
with). The other 26 imatinib records in BindingDB are off-target counter-screen "no
binding" results against unrelated kinases, plus one imatinib-*resistant* ABL1 mutant
(G250E). So feeding wild-type ABL1 asked the model about a sequence/target identity it
effectively had no true-positive training signal for — a sequence-identity mismatch, not
a broken pipeline.

This result, combined with try01, established that TxGemma's zero-shot `BindingDB_ki`
task is fundamentally data-coverage-limited for these targets, motivating the move to
LoRA fine-tuning on additional real data sources (try03 onward).
