# Try 10 — Distribution-based scoring: is the output degeneracy a readout artifact?

**Date:** 23-10-2026
**Scripts:** `distribution_scoring.py` + `distribution_scoring.sb`
**Job:** 23130595 — result pending; see Status below. (23129822 cancelled pre-start, see Status.)

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
| **C — digit-logit expectation** | probability-weighted expected digit per position | Deterministic. Tokenizer confirmed per-digit, so this **does** run |

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

## Result
Job **23130595**, COMPLETED 24-10-2026, elapsed 03:39:25 (inside the 8h budget).

### Validity check passed first
**Method A reproduced try08 exactly: 200/200, 199/199, 200/200, 200/200 identical — 100% on
all four targets.** The comparison is sound; any difference below is the readout, not drift.
Per-digit tokenization confirmed, so Method C ran.

### Degeneracy WAS largely a readout artifact
| Target | most-common-value frequency: greedy → sampled → digit-exp | distinct values |
|---|---|---|
| Gaucher | 791 **51.0% → 27.5% → 5.5%** | 27.5% → 55.0% → **69.0%** |
| Fabry | 630 **24.5% → 2.0% → 1.5%** | 23.0% → 79.0% → **82.5%** |
| TTR | 994 **28.6% → 25.5% → 16.0%** | 47.2% → 63.5% → **68.0%** |
| CF | 727 6.5% → 2.0% → 2.5% | 49.0% → 75.5% → **77.0%** |

Greedy decoding was collapsing these models onto modal outputs. Gaucher's notorious
"791 for half of chemical space" falls to **5.5%** under digit-logit expectation, with no
retraining and no change to the model — the resolution was there all along and the readout
was discarding it. This confirms the hypothesis and vindicates try08's `min_gap = 1`
observation.

**Method C (digit expectation) wins on every target** and is also the cheapest (3 forward
passes, deterministic, no sampling variance). It should become the default scoring method.

### But fixing degeneracy did NOT uniformly improve ranking
Enrichment percentile (lower is better; 50 = chance):

| Target | greedy | sampled | digit-exp | direction |
|---|---|---|---|---|
| Gaucher | 38.5 | 33.0 | **29.0** | improved |
| TTR | 47.7 | 43.5 | **45.0** | ~unchanged, still ≈chance |
| CF | **11.5** | 19.0 | 14.0 | slightly worse |
| Fabry | **15.5** | 74.5 | **83.0** | **catastrophically worse** |

Both pre-registered outcomes occurred, on different targets. That is the honest reading and
it was written down in advance.

### Fabry: the apparent success was entirely a mode artifact
This is the single most consequential number in the run. Greedy gave migalastat **436**,
ranking it in the top 15.5% — which is what made Fabry look like it "passed the category
test" despite failing within-family. But 436 was Fabry's *collapse value*, the mode it
emitted for everything. Integrate over the actual distribution and migalastat's expected
value is **647–660**, ranking **74.5–83.0 percentile — worse than chance**.

Fabry's category-level competence was an accident of its attractor happening to sit at a low
number. Under honest scoring Fabry is the **worst** of the four targets, not the second best.

### The honest ranking after this run
    CF       14.0 percentile   best
    Gaucher  29.0
    TTR      45.0              ~chance
    Fabry    83.0              worse than chance

CF survives as the strongest target on this measure. Note these are the **over-trained,
leak-trained** adapters (try09's epoch-0 models were never saved), so the absolute numbers
inherit those problems; the readout comparison is what this run establishes.

### Bonus: per-molecule uncertainty, and TTR is confidently wrong
Mean sampling standard deviation across 200 molecules (min 0.0, max ~290–325 for all):

    TTR      26.8      lowest variance, yet chance-level ranking
    Gaucher  43.6
    CF       48.4
    Fabry    82.2      highest variance, worst ranking

Variance is **not** a quality proxy — TTR is the most *confident* and ranks at chance, i.e.
confidently wrong, while Fabry is both uncertain and wrong. So uncertainty cannot be used
alone to filter candidates, though Fabry's 82.2 would have flagged it.

### Consequences
1. **Adopt digit-logit expectation as the scoring method everywhere.** Best degeneracy
   reduction on all four targets, cheapest, deterministic. Every historical score in this
   project was a mode, not an estimate.
2. **Every enrichment number recorded before this is suspect**, because they were all
   computed from modes. Fabry's 15.5% → 83.0% is the proof of how badly that can mislead.
3. **Re-run the try09 harness with the new scoring**, for CF first (3 seeds, `N_EPOCH=2`).
   The leak-free numbers in try09 were also greedy-decoded.
4. **Fabry drops to the bottom of the priority list** — it is now the weakest target on the
   only honest measure available, with 89 compounds and a 20% scaffold-holdout cost.
5. Scoring alone did not rescue ranking quality: CF 14%, Gaucher 29% means ~28 and ~58 of
   200 decoys still outrank the real drug. A 10%-shortlist deployment is still not supported.

## Status
COMPLETED. Degeneracy confirmed as largely a greedy-decoding artifact and fixed by
digit-logit expectation (Gaucher 51% → 5.5%). Ranking quality did not uniformly improve, and
Fabry's apparent category-level success was revealed as a mode artifact (15.5% → 83.0%,
worse than chance). Digit expectation adopted as the scoring method going forward.
