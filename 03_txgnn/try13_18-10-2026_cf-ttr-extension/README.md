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
  ~8h each (try11's Fabry/Gaucher took ~7.7h for 3 seeds). Results pending.
