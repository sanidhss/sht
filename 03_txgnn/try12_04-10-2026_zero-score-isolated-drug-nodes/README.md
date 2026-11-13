# try12 — root-cause of the exact-0.0 drug scores (isolated drug nodes)

## What this does
CPU-only diagnostic, no training. `diagnose_zero_scores.py` reads the saved
`disease_eval_[idx]/{train,valid,test}.csv` splits and try11's saved `eval.pkl` files
(`/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection/`) and checks whether
the drugs scoring exactly 0.0 are the drugs with zero edges in the training graph.
Full output: `diagnose_zero_scores.out`.

## Question
Since try10, 3 Gaucher reference drugs — imiglucerase (DB00053), alglucerase (DB00088),
taliglucerase alfa (DB08876) — scored an identical exact 0.0 on every seed. The working
guess was "large protein biologics with missing/degenerate chemistry-derived node
features." **That guess was wrong**: TxGNN's drug node inputs are learned embeddings,
not chemistry features, so there was nothing chemistry-based to be missing.

## Result: confirmed root cause
**Every one of these drugs has exactly one relationship in all of PrimeKG — a single
`indication` edge (stored as 2 rows, forward + reverse). No drug_protein, drug_effect,
drug_drug, or anything else.** For the 3 Gaucher biologics, that one edge is
Gaucher's, which `disease_eval` holds out of training by design. So in the training
graph they are fully isolated nodes (0 edges).

The model then gives an isolated node an all-zero vector, mechanically:
- `HeteroRGCNLayer.forward` / the attention layer (`txgnn/model.py:393-401`,
  `:348-371`) compute a node's new representation **only** from messages from its
  neighbours (`G.multi_update_all(..., 'sum')`), with **no self-loop** — the node's
  own input embedding never feeds into its own output. A node with no neighbours
  receives no messages, and DGL zero-fills it. Two layers of this → still zero.
- The DistMult decoder scores `sum(h_disease * w_rel * h_drug)`, which is exactly 0
  when `h_drug` is the zero vector.

Decisive check, all 6 runs (2 diseases × 3 seeds): **the set of drugs with exact-0.0
scores is identical to the set of drugs with zero training-graph edges** — not a
superset or overlap, identical. Raw scores are signed logits (most drugs score
negative), so 0.0 lands mid-to-low in the ranking (ranks ~5,000–5,950 of 7,957), with
the tied drugs' internal order arbitrary.

| Disease | exact-0.0 drugs | why isolated in train |
|---|---|---|
| Gaucher | imiglucerase, alglucerase, taliglucerase alfa | sole edge = Gaucher indication, held out (in test) |
| Gaucher | tavaborole, racecadotril, metizoline | sole edge landed in the random valid split |
| Fabry | taliglucerase alfa, dornase alfa, iguratimod, mandelic acid, troxipide | sole edge landed in the random valid split |

Contrast that confirms the mechanism: **velaglucerase alfa** (the 4th Gaucher ERT,
same drug class) has one extra `drug_protein` edge in PrimeKG → 2 edges in train →
a real, non-zero, seed-varying score. Being a biologic is irrelevant; graph degree is
what matters.

## What this means for the results
- The 3 biologics' ranks are **not evidence of anything** — not a model failure,
  not a model success. The model cannot score a drug it has never seen connected to
  anything, and under a blind split it is structurally impossible for these 3 to have
  any connection. They are **not evaluable** by any pure graph-topology method on
  PrimeKG as shipped, regardless of seed count or ensembling.
- Gaucher's evaluable reference set is therefore 4 drugs: miglustat, eliglustat,
  velaglucerase alfa, voglibose. Report the 3 biologics separately as
  "not evaluable (no graph edges besides the held-out indication)", not as misses.
- The library's own per-disease metrics (AUROC, Recall@k, "Missed@k") count them as
  misses, so those library summary numbers understate Gaucher slightly. Our own
  `rank_of()`-based reporting is unaffected for evaluable drugs.
- Fabry/migalastat is unaffected (migalastat has normal degree).
- This is a property of the data (PrimeKG's sparse coverage of these biologics) plus
  the architecture (no self-loop), not a bug to patch. Adding a self-loop would only
  give these nodes their randomly-initialized input embedding, which has no training
  signal either — it would replace an exact 0.0 with an equally meaningless random
  score. No library change made.

## For future targets (CF/TTR)
Before training a new disease, run the same degree check on its reference drugs: any
reference drug whose only PrimeKG edges are to the held-out disease will be
unscoreable, and should be flagged up front rather than discovered as a 0.0 afterwards.
