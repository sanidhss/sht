# try13 — extend to cystic fibrosis and TTR amyloidosis (same pure-graph setup)

## What this does
Same model/training as try11 (attention=True, proto, pretrain skipped, indication-AUROC
checkpoint selection via the installed-library patch, `disease_eval` split, 3 seeds),
applied to CF and TTR so the "which disease wins depends on evidence source" pattern can
be tested a third and fourth time. Results go to
`/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection/{cf,ttr}/`.

Files:
- `precheck_degree.py` / `.out` — CPU check, run **before** training (try12's lesson):
  each reference drug's degree in the would-be training graph, plus any drug edges to
  related-named diseases that `disease_eval` would not remove.
- `dryrun_split_check.py` / `.out` — CPU dry run: builds the real splits through
  `TxData.prepare_split`, runs the script's own `check_blind`, and exercises the
  reporting code on try11's saved Gaucher pkls.
- `txgnn_cf_ttr.py` + `txgnn_cf.sb` / `txgnn_ttr.sb` — the training/eval run
  (`DISEASE=cf|ttr`). Both `.sb` files point at **this** folder's script (checked).

Changes vs. try11's script:
- `holdout` is a list (`disease_eval_fold` already accepts one) and `check_blind`
  checks every held-out node.
- Each scored node gets its own `eval_disease_centric` call → `eval_<idx>.pkl`.
- `report_aggregate` computes the **seed-ensembled rank** (mean raw score across seeds,
  then rank) directly, and flags drugs that are exact-0.0 on every seed.

## Target definitions — and the PrimeKG problems found while defining them

### CF — clean
`cystic fibrosis` (internal idx 16250, raw id 9061) is a single node carrying all 10 CF
indications: ivacaftor, lumacaftor, tezacaftor, elexacaftor, dornase alfa,
acetylcysteine, cysteine, mannitol, adenosine phosphate, nafamostat. The other
CF-named nodes (`cystic fibrosis associated meconium ileus`,
`cystic fibrosis-gastritis-megaloblastic anemia syndrome`) have no drug edges, so
holding out 16250 alone is fully blind.

Precheck: the 4 CFTR modulators are well connected in train (440–640 edges each,
drug_drug + 5–13 drug_protein). **Dornase alfa is isolated (train degree 0)** — its
only PrimeKG edge is the CF indication — so it will score exactly 0.0 and is
**not evaluable**, same mechanism as the Gaucher ERTs (try12). Flagged in advance.

### TTR — PrimeKG's labels are sparse and scattered
- `hereditary ATTR amyloidosis` (4803) and `wild type ATTR amyloidosis` (5528) — the
  clinically correct nodes — have **zero** indication edges.
- The only TTR-specific labelled node is `ATTRV122I amyloidosis` (6669), with one
  indication: tafamidis.
- Tafamidis is *also* indicated for `amyloidosis (disease)` (6361), `AL amyloidosis`
  (6665) and `AH amyloidosis` (6008) (the AL/AH links look like PrimeKG annotation
  spread — tafamidis is not an AL therapy — but they are edges all the same).
- Patisiran and inotersen are PrimeKG drug nodes but are not linked to any
  amyloidosis node. Vutrisiran, acoramidis and eplontersen are not in PrimeKG.

**Sibling leakage**: `disease_eval` only removes edges of the listed disease nodes.
Holding out 6669 alone would leave tafamidis→`amyloidosis (disease)` (and AL/AH) in
train — the answer one hop away under a different node name. So the TTR holdout is all
four tafamidis-amyloidosis nodes plus 4803/5528 (the extra two have no drug edges;
included so every node we score against is provably held out). Precheck confirms no
tafamidis→amyloid/transthyretin edge remains in train; tafamidis keeps 171 train edges
(167 drug_drug, 4 drug_protein incl. TTR itself — mechanism, not answer, same status as
migalastat→GLA for Fabry).

Scoring: tafamidis rank at 6669 is the labelled result. Ranks at 4803/5528, and of
diflunisal/patisiran/inotersen anywhere, are **unlabelled in PrimeKG** — informative
(do clinically real ATTR drugs rank high?) but not a hit/miss against ground truth.
Patisiran has only 3 train edges (all drug_protein), so its score rests on very little.
Note TTR has a single labelled drug, so its headline is one rank — weaker evidence than
CF's 10-drug label set.

## Side note found while reading the split code
`TxData.prepare_split` caches the split under `disease_eval_[idx list]/` with no seed in
the path, so seeds 2 and 3 reuse seed 1's 5% validation sample. Pre-existing behavior
(Fabry/Gaucher had it too). Doesn't affect blindness or init/training randomness, but
means validation-set composition is not varied across seeds.

## Status
- **Dry run passed (18-10-2026)** — `dryrun_split_check.out`: both splits built and
  blind (`check_blind` OK on every held-out node). Test DD rows: CF 38 (10 indications
  + 9 contraindications, both directions); TTR 26 (10 indications = tafamidis/
  acetylcysteine/cysteine across the 4 labelled nodes, + 3 contraindications).
  Reporting code reproduces try11's Gaucher ensemble result (miglustat 17/7,957) and
  flags the exact-0.0 drugs correctly.
- **Submitted 18-10-2026: 22996974 (`txgnn_cf`), 22996975 (`txgnn_ttr`)**, 12h each,
  logs at `/work/hdd/bisa/sshah21/results/txgnn/txgnn_{cf,ttr}_<jobid>.out`. Expected
  ~8h each (try11's Fabry/Gaucher took ~7.7h for 3 seeds).
- **Both completed 20-10-2026** (CF 7h48m, TTR 8h28m, exit 0). Blindness check passed
  on all 3 seeds of each. Results below.

## Result — CF and TTR are both weak; neither comes close to Fabry

All ranks are out of 7,957 drugs; ENSEMBLE = mean raw score across 3 seeds, then rank
(adopted method). Chance expectation is ~3,979.

**CF @ cystic fibrosis (16250)**
| drug | per-seed | ensemble |
|---|---|---|
| elexacaftor | 2219 / 1223 / 975 | **1208** |
| tezacaftor | 2392 / 1091 / 1277 | **1301** |
| lumacaftor | 2535 / 1504 / 1238 | **1385** |
| ivacaftor | 2809 / 1556 / 1505 | **1652** |
| cysteine | 4159 / 995 / 1360 | 1601 |
| acetylcysteine | 2170 / 1802 / 2106 | 1853 |
| nafamostat | 3757 / 2896 / 1659 | 2452 |
| adenosine phosphate | 565 / 5614 / 4190 | 3547 |
| mannitol | 6615 / 6865 / 7085 | 6878 |
| dornase alfa | exact 0.0 | not evaluable (predicted by precheck) |

**TTR**
| drug | @ ATTRV122I (6669, labelled) | @ wild-type ATTR (5528) | @ hereditary ATTR (4803) |
|---|---|---|---|
| tafamidis | **974** (982/882/1609) | 980 | 4308 |
| inotersen | 4080 | 3571 | 7151 |
| patisiran | 5156 | 5070 | 3638 |
| diflunisal | 7085 | 7838 | 7884 |

Reading:
- **CF: weak but coherent signal.** All four CFTR modulators land together in the top
  15–21% on every seed, and above chance on every seed — not noise, but ~70–100× worse
  than Fabry's migalastat (17). Mannitol (no protein targets) is near the bottom.
- **TTR: tafamidis top ~12%** at the one labelled node, and similar at wild-type ATTR.
  Single labelled drug, so this is one data point. Hereditary ATTR (4803) shows no
  usable signal — every drug's score there is ~0 (tafamidis mean −0.0002), consistent
  with a near-empty node (see below).
- **Exact-0.0 drugs were exactly the isolated ones again**: CF dornase alfa (predicted)
  + mandelic acid (sole edge landed in valid); TTR 4 drugs, same mechanism (try12).
- Model training itself was fine: best validation indication AUROC CF 0.823, TTR 0.826
  (Fabry 0.815, Gaucher 0.810). This is a per-disease placement result, not a failed run.

**Library AUROC is NOT comparable to our rank metric.** `calculate_metrics`
(`txgnn/utils.py` ~L1209) restricts AUROC to `fixed_keys` = drugs that have an
indication somewhere in PrimeKG (~1–2k drugs), not all 7,957. So CF's library AUROC
(0.92/0.87/0.88) and TTR-6669's (0.93/0.97/0.92) look strong while the full-library
ranks are mediocre. Report the full-library rank; it's the stricter and the one used
for Fabry/Gaucher.

## Diagnosis so far — what is and isn't the explanation

**Disease-node connectivity in train** (from the split CSVs):
| node | train edges | of which disease_protein |
|---|---|---|
| Fabry (543) | 218 | 23 |
| Gaucher (16430) | 266 | 5 |
| CF (16250) | 90 | 11 |
| ATTRV122I (6669) | **6** | 1 (TTR) |
| wild-type ATTR (5528) | 50 | 0 |
| hereditary ATTR (4803) | **4** (all disease_disease) | 0 |

Hereditary ATTR is almost empty — not even linked to the TTR gene — which explains its
~0 scores. ATTRV122I is extremely sparse (6 edges).

**Shared-target path (drug→protein←disease) is NOT the missing ingredient.** It exists
for every relevant case: migalastat→GLA←Fabry (rank 17), CFTR modulators→CFTR←CF
(~1,200–1,650), tafamidis→TTR←ATTRV122I (974), diflunisal→TTR←ATTRV122I (7,085!).
Gaucher's miglustat has *no* shared target with Gaucher yet reaches 17 ensembled. So
having the mechanistic path neither guarantees nor is required for a good rank.

**Unverified hypothesis (not tested):** dilution by the drug's other neighbours. The
CFTR modulators each also target promiscuous metabolism/transport hubs (CYP3A4, ABCB1,
ALB, several CYP2C*) and carry 435–627 drug_drug edges; diflunisal (worst TTR rank
despite sharing TTR) has 1,538 drug_drug edges and an NSAID target profile; tafamidis
(171 edges) does better than diflunisal. But migalastat also had ~770 drug_drug edges
(try07) and still ranks 17, so this is at best partial. Testing it would need e.g.
attention-weight inspection on the CFTR/TTR paths, or the same comparison over many
diseases — not done.

## Cross-track status
TxGemma-predict's leak-free re-evaluation (its try09) changed its own CF/TTR picture:
TTR passes 2/3 seeds leak-free; its earlier CF pass did not reproduce on the one seed
that finished (CF timed out, re-run pending). So the clean "evidence source decides
the winner" contrast seen for Fabry/Gaucher **cannot yet be claimed for CF/TTR** —
TxGNN is weak on both, and TxGemma-predict's CF result is unsettled. Revisit once
TxGemma-predict CF completes.
