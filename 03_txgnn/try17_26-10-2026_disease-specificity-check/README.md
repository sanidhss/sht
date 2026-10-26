# try17 — is the ranking disease-specific? (Mostly no.)

## Question
Prompted by asking whether the model also suggested migalastat for Gaucher. CPU only:
`specificity_check.py` / `.out` compares the seed-ensembled score lists from try11
(Fabry, Gaucher) and try13 (CF, TTR @ ATTRV122I).

## Result
**The model produces two near-fixed rankings, not one per disease.**

| | Fabry–Gaucher | CF–TTR | Fabry/Gaucher vs CF/TTR |
|---|---|---|---|
| Spearman over all 7,957 drugs | **0.957** | **0.926** | 0.29–0.39 |
| top-50 overlap | **41/50** | 29/50 | **0/50** |

- Fabry and Gaucher get a "rare genetic / lysosomal" list. Its top 10 is
  enzyme-replacement therapies for *other* lysosomal diseases (idursulfase,
  galsulfase, elosulfase alfa, vestronidase alfa: MPS II/VI/IVA/VII) plus generic
  entries (avelumab, selumetinib, roxadustat, mecasermin, siltuximab).
- CF and TTR get a "corticosteroid / chemotherapy" list (prednisone, dexamethasone,
  methylprednisolone, cisplatin, carboplatin, ...).

**Cross-disease ranks** (rows are each disease's true drugs, columns are the disease
they're scored for):

| drug | Fabry | Gaucher | CF | TTR |
|---|---|---|---|---|
| migalastat (Fabry) | **16** | 27 | 7341 | 7400 |
| miglustat (Gaucher) | 60 | **17** | 7778 | 7904 |
| eliglustat (Gaucher) | 15 | **921** | 7379 | 7488 |
| velaglucerase alfa (Gaucher) | 19 | **3423** | 7601 | 7672 |
| elexacaftor (CF) | 61 | 93 | **1208** | 7548 |
| ivacaftor (CF) | 69 | 131 | **1652** | 7591 |
| tafamidis (TTR) | 46 | 48 | 7263 | **974** |

## What this means
- **Fabry's rank of 16 is mostly not Fabry-specific.** Fabry's list ranks almost every
  rare-disease drug highly: the Gaucher drugs at 15–60, tafamidis at 46, and the CF
  drugs at 61–69. The CF drugs rank **~20× better for Fabry than for CF**. Migalastat
  is only marginally more "Fabry" than "Gaucher" (16 vs 27). The model is mostly
  recognising "rare genetic disease → rare-disease drugs", not "Fabry → migalastat".
- **Gaucher's 17 has the same caveat**, plus the Niemann-Pick sibling path (try14/15).
  Eliglustat and velaglucerase alfa actually rank *better for Fabry* than for Gaucher.
- **CF/TTR look weak partly because they land in the other mode.** Their true drugs
  are rare-disease drugs, but their disease nodes get the steroid/chemo ranking. The
  question becomes why CF/TTR's nodes resemble common inflammatory/oncology diseases
  to the model. That's not investigated yet; it may relate to their sparse,
  generic-phenotype neighbourhoods (try13: CF 45 non-drug edges, ATTRV122I 3).
- This matches the TxGemma-chat track's try08 finding ("class-level screening works,
  within-class discrimination does not"): a second, independent model showing
  category-level rather than within-category signal.

## Consequences
- Fabry/Gaucher should be reported as **category-level success**: the right drug class
  rises to the top of the rare/lysosomal list. They shouldn't be reported as
  disease-specific identification.
- Every rank reported from now on should come with a specificity control: how the same
  drugs rank for other diseases. try16's `analyze.py` was updated (before its job ran)
  to compute `specificity` = median percentile of a disease's true drugs in other
  diseases' rankings minus in its own ranking.
- Unverified: whether the two "modes" are a general property (a few dominant ranking
  templates across all diseases) or specific to these four. try16's 120 diseases will
  show this (pairwise correlation of their rankings).
