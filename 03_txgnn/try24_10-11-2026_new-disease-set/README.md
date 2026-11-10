# try24 — six new diseases on the augmented graph (fixed try23 config)

## What
Runs TxGNN on six new rare diseases with exactly the try23 "augmented + symptom
lookalikes" setup, as one Slurm chain. try23's files are untouched.

| key | disease | reference drug |
|---|---|---|
| cel | chronic eosinophilic leukemia | imatinib |
| sm | systemic mastocytosis | midostaurin (also tracks avapritinib) |
| nsiad | nephrogenic syndrome of inappropriate antidiuresis | tolvaptan |
| pku | phenylketonuria | sapropterin |
| mh | malignant hyperthermia | dantrolene |
| tyr1 | tyrosinemia type 1 | nitisinone |

## Fixed config (not swept, not changed after seeing any new-disease result)
`DATA_DIR=/work/hdd/bisa/sshah21/primekg_aug`, `sim_measure='all_nodes_profile_more'`
(4-type), n_hid/n_inp/n_out=100, proto=True, proto_num=3, attention=True,
agg_measure='rarity', pretrain() skipped, finetune(n_epoch=500, lr=5e-4), seeds 1, 2, 3.
Results go to `/work/hdd/bisa/sshah21/results/txgnn_v4_new_disease_set/`.

**Why this signature:** `sim_measure='all_nodes_profile_more'` was chosen on try23's
results for Fabry/Gaucher/CF/TTR (better rank on 5 of 6: Fabry 3170→486, TTR wild-type
548→138, TTR hereditary 4945→592), and fixed before any new-disease run. It is also the
signature the TxGNN paper specifies (4 node types; see try20). Choosing it after seeing
the new ranks would be selecting the method on the outcome.

No library changes. The installed, patched `txgnn` is used as-is (no pip install).

## Files
- `config.py`: every id, holdout list, drug and path. The single source for all jobs.
- `resolve_and_audit.py` (job A, CPU): resolves ids and builds the splits. Checks
  blindness and the prototype path. Writes `resolution_report.md`, `resolved.json` and
  `audit.out`.
- `txgnn_newset.py` (job B, GPU array, one task per disease): training and scoring.
- `collate.py` (job C, CPU): writes `results.csv`, `results_ensemble.csv` and `SUMMARY.md`.
- `A_resolve_audit.sb`, `B_train.sb`, `C_collate.sb`, `submit_chain.sh`.

## How job A decides (rules fixed before any training)
- **IDs:** prints every name match with its internal index, degree and phenotype-edge
  count. It fails if a configured node or drug doesn't resolve, or if a name differs
  from what's expected.
- **Evaluable:** the primary drug must have an indication or off-label edge to the scored
  node. If not, the disease is reported as not evaluable and its array task exits
  without training. That alone doesn't fail the chain.
- **Holdouts:** any disease carrying a reference drug must be held out if it is a MONDO
  neighbour (parent/child) of the scored node, or has cosine ≥ 0.5 to it on the 4-type
  profile. Otherwise job A fails.
- **Blindness:** no drug–disease edge of any held-out node may be in train.
- **Prototype path (try18):** the 3 lookalikes are rebuilt from the real training split
  with the 4-type signature. If any lookalike carries a reference drug among its
  training indications, job A fails. No sibling exceptions are allowed here (unlike
  Gaucher→Niemann-Pick in try23).
- **Splits:** `disease_eval` splits are cached per holdout list, not per seed (checked in
  `TxData.prepare_split`). All 3 seeds share one split, so the audit is exact for every
  seed. Seeds differ only in model initialisation and training.

## Step 1–2 findings (exploratory job A runs, 10-11-2026, before any training)
- **Avapritinib:** no systemic-mastocytosis edge in PrimeKG (its only indication is GIST).
  Not evaluable for SM; its rank is shown for information only. Midostaurin is the primary.
- **Tyrosinemia type 1:** PrimeKG has no type-1 node. The only match is a merged node
  "tyrosinemia" (`10162_10160_10161_4741`, types 1–3 folded together). It is the single
  candidate, so it is used, with this caveat. It carries a nitisinone indication *and* a
  contraindication. Both are held out with the node.
- **NSIAD:** matches only "nephrogenic syndrome of inappropriate antidiuresis" (10356),
  which has a tolvaptan indication. Caveat: NSIAD is caused by an activating AVPR2
  mutation that vaptans generally do not block, so this PrimeKG edge is clinically
  questionable (likely confused with SIADH). PrimeKG has no separate SIADH node except
  "inappropriate ADH syndrome" (6802, tolvaptan). It is held out to be safe, though its
  cosine is 0 and it is not a MONDO neighbour.
- **Chronic eosinophilic leukemia (the known risk):** there are two CEL nodes: 15687 (genes
  FIP1L1, PDGFRA) and 6147 "CEL, not otherwise specified". Both carry imatinib, so **both
  are held out and both are scored** (15687 primary). First audit: with only those two
  held out, CEL's lookalikes #1 and #3 were *hypereosinophilic syndrome* (15691) and
  *idiopathic hypereosinophilic syndrome* (11895). Both carry imatinib, and idiopathic HES
  has exactly CEL's genes (FIP1L1, PDGFRA). FIP1L1-PDGFRA "HES" is the same disease
  entity (WHO: myeloid neoplasm with PDGFRA rearrangement). They were the same kind of
  leak as Fabry's chylomicron retention disease, so **both HES nodes are held out too**.
  CEL is kept, not dropped. The CEL nodes have no phenotype edges and only 2 genes, so
  their lookalikes rest on very little data.
- **Systemic mastocytosis:** first audit: lookalikes *aleukemic mast cell leukemia* (cos
  0.50) and *mast cell leukemia* both carried midostaurin. The parent *mastocytosis*
  (7950) and the subtypes *aggressive SM* (20333), *mast cell leukemia* (20334),
  *aleukemic MCL* (15561) and *obsolete classic MCL* (15560) all carry midostaurin, so all
  are held out.
- **Phenylketonuria:** parent *disorder of phenylalanine metabolism* (17306) carries
  sapropterin, so it is held out.
- **Malignant hyperthermia:** two nodes carry dantrolene: *MH of anesthesia* (18493) and
  *exercise-induced MH* (18752). The second was the first audit's #1 lookalike for the
  first. Neither is "the" MH node, so both are held out and both are scored (18493
  primary). *Neuroleptic malignant syndrome* has dantrolene only as off-label, which is
  not on the prototype path.
- **Reliability (first audit, before the extra holdouts):** NSIAD (0.18), PKU (0.18) and MH
  (0.17) had best lookalike cos < 0.2. If that holds in the final audit, they are
  reported as UNRELIABLE, not as ranks.

The final audit with these holdouts is `audit.out` / `resolution_report.md`.

## Job chain
See the log below for job ids.

## Log
- 10-11-2026: submitted chain A=23702737 (CPU audit) -> B=23702757 (GPU array 0-5: cel sm nsiad pku mh tyr1, afterok:A) -> C=23702758 (collate, afterany:B). Directory frozen from here.
