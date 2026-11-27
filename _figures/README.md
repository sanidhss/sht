# Figures

**Not committed to git.** This directory is untracked by deliberate decision
(25-11-2026): the side-by-side percentiles are held locally, and each arm's own numbers
stay in its track README.

## equalized_pool_blinded_gnn_vs_gemma.{png,pdf,csv}

Blinded percentile of the matched reference drug among the 1,933 drugs of `pool_v1`
(`06_equalized_pool/try01`, sha256 `9d89c3ce…`), lower is better, chance = 49.902%.
Three series: TxGNN (graph), TxGemma-predict (transformer, post-LoRA), and a Tanimoto
5-NN chemical baseline. Regenerate with:

```
/sw/external/python/anaconda3/bin/python3 -I make_fig.py
```

(The project conda envs have no matplotlib except `envs/txgemma`.)

Sources, all read-only: `03_txgnn/try25_23-11-2026_equalized-pool/` for the ensembled
ranks plus specificity columns and the per-seed file beside it, and
`try16_equalized_pool/results_pool_v1.csv` under `$WORK` for the transformer and the kNN.

### What the first version of this figure got wrong

It plotted two bars and nothing else, which let three false readings through. All three
are now shown on the figure itself.

**1. The chemical baseline was missing, and it beats both models outright twice.** The
kNN percentile was sitting in the `tanimoto_percentile_pool` column of the very file the
TxGemma bars were read from, and was ignored. It is now a red diamond on a vertical rule
through the row:

| disease | TxGNN | TxGemma (LoRA) | Tanimoto 5-NN |
|---|---|---|---|
| Fabry | 6.1 | 40.5 | 85.8 **trivial, inadmissible** |
| ATTR amyloidosis | 8.0 | 68.6 | **57.3** (beats the transformer) |
| Cystic fibrosis | 15.0 | 38.9 | 71.0 |
| Tyrosinemia | 29.3 | 48.8 | **18.4 — beats both** |
| Chronic eos. leukemia | 43.4 | 10.3 | 17.8 |
| Systemic mastocytosis | 50.5 | 63.3 | **0.6 — beats both, by ~100x** |

Mastocytosis is the row that matters: stereo-blind Morgan-2 fingerprints with k=5 put
midostaurin **11th of 1,933** while both deep models sit at chance.

The kNN is a property of the disease, pool and actives list rather than of either arm, so
one marker is the baseline for both bars on its row. It needs measured bioactivity
against the target, which exists for 6 of the 10 diseases: MH and PKU have no actives set
at all (never TxGemma targets — see `pool_common.ALL_TARGETS`), while gaucher and nsiad
have one but cannot rank a reference that is absent from the pool. The CSV distinguishes
`NO_ACTIVES_SET` from `REFERENCE_NOT_IN_POOL`.

**2. A low TxGNN bar was being read as a disease-level hit.** try25's `specificity_pass`
column asks whether some *other* disease's model ranks the same drug better. Six of the
nine rankable nodes fail. Failing bars are now hatched and the margin column names the
competing rank:

| disease | TxGNN rank | best rank under another disease's model | |
|---|---|---|---|
| Gaucher / miglustat | 10 | 27 (fabry) | pass |
| MH / dantrolene | 11 | 29 (sibling MH node) | pass |
| PKU / sapropterin | 211 | 778 (fabry) | pass, but `reliable=False` |
| Fabry / migalastat | 118 | **17 (cf)** | FAIL |
| ATTR / tafamidis | 155 | **8 (fabry)** | FAIL |
| CF / lumacaftor | 289 | **72 (fabry)** | FAIL |
| Tyrosinemia / nitisinone | 566 | **34 (gaucher)** | FAIL |
| CEL / imatinib | 839 | **121 (cf)** | FAIL |
| Mastocytosis / midostaurin | 976 | 768 (fabry) | FAIL |

So the arm's good-looking percentiles are better than chance but **not disease-specific**:
the Fabry model ranks tafamidis 19x better than TTR's own model does. "Better than chance"
and "a hit" are different claims, and the percentile axis cannot tell them apart.

Two corrections this table forces on text written earlier, including an earlier version
of this file:

- Gaucher **passes** the specificity control (10 vs 27). The claim that TxGNN fails it on
  everything except malignant hyperthermia is wrong.
- By `specificity_pass` alone there are three passes (MH, Gaucher, PKU), not one;
  `03_txgnn/README.md`'s "one clean pass and one borderline" drops PKU on the separate
  `reliable` flag. That criterion combines two tests without saying so, and should be
  stated as two.

**3. The TxGemma bars are post-LoRA, and that was not stated.** try16's CSV has no `arm`
column — every row is a fine-tuned adapter (blinded, `digit_exp`, 3 seeds). No pre-LoRA
bar can be drawn, because zero-shot was only ever measured on the old 8-target decoy pool.
On that field LoRA moved mastocytosis 14.6 → 77.9 and cel 94.2 → 20.6, so some green bars
here are far *worse* than the untrained base model and some far better. The legend now
says AFTER LoRA.

### Two bars, two different statistics

TxGNN's bar is the rank of the **mean raw score** over seeds 1-3 (try11's adopted
ensembling, chosen because the per-seed variance is inherent embedding stochasticity
rather than a fixable training bug); TxGemma's is the **mean of three per-seed
percentiles**. Both whiskers are the per-seed min-max, so TxGNN's is a spread around a
differently-computed centre and need not be centred on its bar. Gaucher is the clearest
case: it ensembles to 0.5 while its seeds land at 0.2 / 0.7 / 17.5 (ranks 3, 13, 339).

Per-seed spread is large on both arms and is the main reason few gaps here are assertable.
TxGNN's widest are tyrosinemia (7.7-43.0) and Gaucher (0.2-17.5), tightest MH (0.6-1.1).
TxGemma's widest are tyrosinemia (17.8-70.1) and CF (14.5-58.4), tightest cel (7.5-12.2).

### Two kinds of blank

- **no TxGemma arm** — MH and PKU are TxGNN-only; the transformer never ran them.
- **reference not in pool** — the arm exists but the drug is absent from `pool_v1`:
  ambroxol is not `max_phase=4` in ChEMBL (Gaucher), and tolvaptan's UniChem partner is a
  different ChEMBL record (NSIAD). Nothing was substituted. For Gaucher, TxGNN scores
  miglustat instead, which is a different question from the one the thesis asks.

Where TxGNN scores several drugs or nodes for one disease (CF's 4 modulators, Gaucher's 2,
TTR's 3 nodes, CEL's 2), the figure plots the one matching TxGemma's reference drug, and
for TxGNN-only rows the primary node. ATTR is the ATTRV122I node, TTR's only labelled one.

### Still not a scoreboard

Even with the baseline and the specificity control drawn, this omits the unblinded
ceilings (01/try17, 03/try26, both queued), so it shows where each arm lands without
showing how well it could possibly do. And the kNN carries its own leak control: Fabry's
baseline is `trivial_flag=True` (Tanimoto ≥ 0.95 to a training active), drawn dotted, and
must not be counted as a comparison.
