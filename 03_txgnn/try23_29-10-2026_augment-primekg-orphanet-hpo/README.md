# try23 — augment PrimeKG with Orphanet (disease→gene) + HPO (disease→symptom)

## Why
try18/try20 showed the thesis diseases are limited mainly by how they're represented in
PrimeKG: Fabry has a mislabelled duplicate (chylomicron retention disease) carrying its
answer, and the TTR nodes are nearly empty. This builds a corrected, rare-disease-enriched
copy of PrimeKG. The original stays untouched.

## Design rules (`build_augmented_kg.py`)
- **Only adds edges between nodes that already exist in PrimeKG.** TxGNN numbers nodes by
  sorting their IDs, so an unchanged node set keeps every index (Fabry 543, etc.) and all
  existing scripts valid.
- **Never adds drug–disease edges** (those are the answers). Only genes and symptoms are added.
- Sources:
  - Orphanet product 6: gene associations with status "Assessed", excluding "Candidate gene
    tested in" and "Biomarker tested in".
  - HPO `phenotype.hpoa`: aspect P, excluding NOT.
  - IDs are mapped via MONDO cross-references, using only unambiguous ones (an Orphanet/OMIM
    ID that maps to exactly one MONDO term).
- PrimeKG predates the current MONDO release, so some of its disease terms are now obsolete.
  An obsolete term inherits its replacement's cross-references. Example: hereditary ATTR
  amyloidosis MONDO:0017132 → 0007100, which PrimeKG also has as "familial amyloid
  neuropathy", so both nodes get the data.
- **Fabry-duplicate fix:** remove migalastat ↔ chylomicron retention disease, and that
  disease's gene links duplicated from Fabry (22, incl. GLA). It keeps SAR1B, APOC2 and LPL.
- `--dry-run` writes only `report.txt`, `added_edges.csv` and `removed_edges.csv`.

## Dry run on Delta (29-10-2026; Orphanet/HPO/MONDO downloaded the same day, HPO release 2026-09-01)
Added 80,710 rows (both directions): 3,748 disease→gene and 76,962 disease→symptom.
Removed 46 rows (the fix). For the thesis diseases:

| disease | genes before→after | symptoms before→after |
|---|---|---|
| Fabry | 23→24 | 69→78 |
| Gaucher | 5→5 | 115→149 |
| CF | 11→25 | 32→51 |
| ATTRV122I amyloidosis | 1→1 | 0→21 |
| hereditary ATTR amyloidosis | 0→0 | 0→34 |
| wild type ATTR amyloidosis | 0→0 | 23→35 |
| chylomicron retention disease | 25→3 | 26→26 |

Limit: hereditary ATTR still gets no TTR gene. Orphanet attaches genes to subtypes (e.g.
ATTRV30M, ATTRV122I), not to the umbrella term. Not hand-added, to keep every edge
source-derived.

Note: our current similarity setting (`all_nodes_profile`) uses only genes + related
diseases to pick lookalikes. The added symptoms still change every node's learned
representation, but they affect lookalike choice only under `all_nodes_profile_more`.

## Revisions after review (29-10-2026)
Built on Delta via Slurm (CPU), not locally.

| review point | how it's handled |
|---|---|
| MONDO as disease layer | ORPHA/OMIM mapped via MONDO xrefs (unambiguous only); no new nodes |
| enrich, don't duplicate | existing disease–gene / disease–phenotype edges skipped |
| edge attributes | filtered rather than new relation types: HPO drops IEA (electronic-only) and very rare / excluded / <5% frequency |
| few relation types | Orphanet keeps only `Disease-causing*` (causal) associations, in PrimeKG's existing `disease_protein` type |
| leakage | no drug–disease edges added. Fabry also holds out its parent category *lysosomal lipid storage disorder* (6521), whose migalastat link was kept out of the baseline only by luck. `verify_augmented.py` fails the chain on any blindness failure or new lookalike carrying a true drug (Gaucher → Niemann-Pick allowed as a known legitimate sibling) |
| pipeline / versions | appended in PrimeKG's exact format (not the full PrimeKG rebuild, which would renumber every node); HPO, Orphadata and MONDO versions are written to `build_report.txt` |
| fair evaluation | same holdouts and model config as baseline; `--only genes|phenotypes` ablation switch exists, combined graph run first |

## Job chain (submitted 29-10-2026)
1. `build.sb` → **23321609** (CPU): builds `/work/hdd/bisa/sshah21/primekg_aug/kg.csv`, links
   the unchanged node/edge tables, and copies the report here as `build_report.txt`.
2. `verify.sb` → **23321610** (CPU, after build): builds TxGNN's processed graph, checks every
   node index equals the original, builds the 4 thesis splits in parallel, and runs the
   blindness and lookalike checks. Output in `verify.out`. Exits non-zero on any problem.
3. `train_aug.sb` × 4 (GPU, only if verify passes): **23321611** fabry, **23321612** gaucher,
   **23321613** cf, **23321614** ttr. `txgnn_aug.py` is the try13 script, pointed at the
   augmented graph with results in `txgnn_v3_augmented/`.

Compare against the baseline runs (try11/try13) with the same seed-ensembled ranks.


## 30-10-2026
- Build 23321609 done (73,192 rows added with filters; see build_report.txt).
- Verify 23321610 FAILED: TxData() does not create kg_directed.csv (prepare_split does). Fixed by calling preprocess_kg directly. GPU jobs were auto-cancelled by the failed dependency.
- Resubmitted: verify 23366266, then fabry 23366267, gaucher 23366268, cf 23366269, ttr 23366270.
