# Try 10 — Distribution-based scoring: is the output degeneracy a readout artifact?

**Date:** 23-10-2026
**Scripts:** `distribution_scoring.py` + `distribution_scoring.sb`
**Job:** 23129822 — result pending; see Status below.

**No training.** Reuses the four already-saved adapters and try08's exact molecule set, and
changes only *how* a score is read out of the model.

## The hypothesis
Every score this project has produced came from one line:

```python
out = model.generate(**ids, max_new_tokens=8)
```

No sampling argument, so transformers defaults to `do_sample=False` — plain **greedy
decoding**, argmax at each token. For a model emitting a number as text, greedy decoding
returns the **modal** output, not an estimate.

That is exactly the shape of the degeneracy that survived every eval fix in try09:

| Adapter | most common value on 200 novel compounds |
|---|---|
| Gaucher | **791 × 102 (51%)** |
| TTR | 994 — and ~40% of a separate 500-decoy set |
| Fabry | 630 × 49 (25%) |
| CF | 727 × 13 (6.5%) — the one that looks healthy |

Those may not be predictions. They may be modes.

**Reason to think the capability is there and the readout is throwing it away:** try08
measured `min_nonzero_gap = 1` for all four adapters. The models demonstrably *can* produce
fine-grained values. If degeneracy were a hard capacity limit, that gap would be large.

## What it does
Three scoring methods, same models, same molecules. The only variable is the readout.

| Method | What it does | Notes |
|---|---|---|
| **A — greedy** | current method | Also a **reproduction check** against try08 |
| **B — sampling mean** | K=16 completions at T=0.8, parse, average | Primary. Tokenization-agnostic |
| **C — digit-logit expectation** | probability-weighted expected digit per position | Deterministic; only runs if the tokenizer splits numbers per-digit, checked at runtime |

**Method A doubles as a validity check.** If greedy here doesn't reproduce try08's recorded
values on the same molecules and adapters, something else is wrong and the comparison is
void. The script prints the agreement rate rather than assuming it.

**Why sampling is primary:** it makes no assumption about how the tokenizer splits numbers,
so a tokenization surprise can't silently invalidate it. Method C is cheaper and
deterministic but depends on per-digit tokenization, so it is attempted defensively and
skipped if that assumption fails.

**Free bonus from sampling:** the standard deviation across samples is a per-molecule
**uncertainty** estimate. That is independently useful — a candidate at 400 ± 15 is a
different proposition from 400 ± 200, and the current pipeline cannot tell them apart.

## Which adapters, and the limitation that comes with it
The four **original** per-target adapters — the over-trained (3-epoch), leak-trained models.
**Not** try09's epoch-0 models, which were never saved. That was an oversight in try09: it
wrote summaries but never called `save_pretrained`.

It turns out to be the right choice anyway, because try08 measured the degeneracy on exactly
these adapters, so greedy-vs-distribution on the same weights isolates the readout change
perfectly. But it does mean the absolute numbers here inherit the over-training problem, and
the conclusion transfers to a properly-stopped model only by assumption.

## Pre-registered reading of the result
Written before the run so it cannot be read selectively afterwards.

| Outcome | Meaning |
|---|---|
| Top-value frequency drops sharply (Gaucher 51% → single digits) **and** enrichment holds or improves | The degeneracy was a **readout artifact**. Cheap, substantial win — and it would mean every historical score in this project understated the model's resolution. |
| Top-value frequency drops **but** enrichment gets worse | Ties were being broken in the drug's favour by luck. The ranking signal is weaker than the greedy numbers implied. Useful, unwelcome. |
| Top-value frequency barely moves | The degeneracy is **real in the model**, not the readout. The next lever is then training-side — mid-epoch stopping, lower LR — not scoring. |

## Why this was prioritised over the alternatives
- **Over re-running CF** (also needed, and still queued behind this): CF is one job that
  yields one verdict. This is one job that could change how *every* score in the project is
  computed, and it runs on data already on disk.
- **Over more training data:** volume was disproved by try07.
- **Over 27B:** the failures aren't capacity-shaped — 89- and 493-compound targets fail while
  1,208 and 1,427 pass on the same model — and 27B in bf16 (~54GB) exceeds the 42GB card.
- **Over the RDKit structural-descriptor phase:** scoped for premises that no longer hold.

## Caution, so this isn't oversold
Better scoring fixes **ties**. If the underlying ranking signal is weak, breaking ties won't
make the right drug come first — it will spread a weak ranking across more values. Enrichment
could go from 20% to 12%, or barely move. This is the best-expected-value cheap experiment,
not a guaranteed fix.

## Status
Submitted as job **23129822** (8h budget, gpuA100x4, 1 GPU, inference only). Result pending.

Confirmed before submission: try08's saved raw generations contain the exact 200-molecule
novel stratum and 9 eval anchors per target, so the comparison is on identical inputs.
