# try24 resolution report


## cel: chronic eosinophilic leukemia

| raw id | internal idx | name | degree | phenotype edges |
|---|---|---|---|---|
| 11895 | 1502.0 | idiopathic hypereosinophilic syndrome | 80 | 76 |
| 15691 | 3582.0 | hypereosinophilic syndrome | 25 | 0 |
| 17834 | 5367.0 | secondary hypereosinophilic syndrome | 2 | 0 |
| 15687 | 3577.0 | chronic eosinophilic leukemia | 4 | 0 |
| 6147 | 13744.0 | chronic eosinophilic leukemia, not otherwise specified | 5 | 0 |
| 17833 | 5366.0 | primary hypereosinophilic syndrome | 1 | 0 |
| 17835 | 5368.0 | lymphocytic hypereosinophilic syndrome | 1 | 0 |
| 100059 | 46.0 | hypereosinophilia of undetermined significance | 1 | 0 |

Causal/associated genes of eval and held-out nodes (augmented graph):

- chronic eosinophilic leukemia (15687): 2 genes: FIP1L1, PDGFRA
- chronic eosinophilic leukemia, not otherwise specified (6147): 2 genes: FIP1L1, PDGFRA
- hypereosinophilic syndrome (15691): 12 genes: ETNK1, FIP1L1, GATA3, IL5, IL6ST, JAK1, MYD88, PDGFRA, RAG2, SHARPIN, TLR2, TLR4
- idiopathic hypereosinophilic syndrome (11895): 2 genes: FIP1L1, PDGFRA

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Imatinib | DB00619 | 15687, 6147 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| hypereosinophilic syndrome | 15691 | Imatinib | 0.246 | False | True |
| idiopathic hypereosinophilic syndrome | 11895 | Imatinib | 0.130 | False | True |
| gastrointestinal stromal tumor | 11719 | Imatinib | 0.070 | False | False |
| mastocytosis | 7950 | Imatinib | 0.000 | False | False |
| blast phase chronic myelogenous leukemia, BCR-ABL1 positive | 6115 | Imatinib | 0.000 | False | False |
| systemic mastocytosis | 16586 | Imatinib | 0.000 | False | False |
| chronic myelogenous leukemia, BCR-ABL1 positive | 11996 | Imatinib | 0.000 | False | False |
| dermatofibrosarcoma protuberans | 11934_23273 | Imatinib | 0.000 | False | False |

## sm: systemic mastocytosis

| raw id | internal idx | name | degree | phenotype edges |
|---|---|---|---|---|
| 7950 | 15301.0 | mastocytosis | 48 | 26 |
| 19314_19023_2726 | 6562.0 | cutaneous mastocytosis | 35 | 27 |
| 19315_17243 | 6563.0 | diffuse cutaneous mastocytosis | 32 | 28 |
| 19316 | 6564.0 | maculopapular cutaneous mastocytosis | 34 | 27 |
| 20333 | 7351.0 | aggressive systemic mastocytosis | 66 | 56 |
| 20331 | 7349.0 | indolent systemic mastocytosis | 29 | 24 |
| 20332 | 7350.0 | systemic mastocytosis with an associated clonal hematologic non-mast cell lineage disease | 47 | 42 |
| 17244 | 4898.0 | pseudoxanthomatous diffuse cutaneous mastocytosis | 4 | 1 |
| 20334 | 7352.0 | mast cell leukemia | 5 | 0 |
| 16586 | 4301.0 | systemic mastocytosis | 21 | 0 |
| 15560 | 3478.0 | obsolete classic mast cell leukemia | 2 | 0 |
| 15561 | 3479.0 | aleukemic mast cell leukemia | 3 | 0 |
| 15557 | 3475.0 | Smouldering systemic mastocytosis | 2 | 0 |
| 15558 | 3476.0 | isolated bone marrow mastocytosis | 2 | 0 |
| 15559 | 3477.0 | lymphoadenopathic mastocytosis with eosinophilia | 1 | 0 |

Causal/associated genes of eval and held-out nodes (augmented graph):

- systemic mastocytosis (16586): 2 genes: HMOX1, KIT
- mastocytosis (7950): 3 genes: FOXF1, HMOX1, KIT
- aggressive systemic mastocytosis (20333): 7 genes: ASXL1, CBL, HMOX1, KIT, RUNX1, SRSF2, TET2
- mast cell leukemia (20334): 1 genes: KIT
- aleukemic mast cell leukemia (15561): 1 genes: KIT
- obsolete classic mast cell leukemia (15560): 1 genes: KIT

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Midostaurin | DB06595 | 16586 |
| Avapritinib | DB15233 | NONE - not evaluable for this disease |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| aleukemic mast cell leukemia | 15561 | Midostaurin | 0.500 | False | True |
| obsolete classic mast cell leukemia | 15560 | Midostaurin | 0.354 | False | True |
| mast cell leukemia | 20334 | Midostaurin | 0.177 | True | True |
| mastocytosis | 7950 | Midostaurin | 0.120 | True | True |
| aggressive systemic mastocytosis | 20333 | Midostaurin | 0.088 | True | True |
| gastrointestinal stromal tumor | 11719 | Avapritinib | 0.043 | False | False |
| acute myeloid leukemia with NPM1 somatic mutations | 18437 | Midostaurin | 0.027 | False | False |
| acute myeloid leukemia with inv3(p21;q26.2) or t(3;3)(p21;q26.2) | 18435 | Midostaurin | 0.027 | False | False |
| acute myeloid leukemia with t(9;11)(p22;q23) | 18434 | Midostaurin | 0.027 | False | False |
| acute myeloid leukemia with t(6;9)(p23;q34) | 18433 | Midostaurin | 0.027 | False | False |
| acute myeloid leukemia with CEBPA somatic mutations | 17894 | Midostaurin | 0.027 | False | False |
| acute myeloid leukemia with t(8;21)(q22;q22) translocation | 15166 | Midostaurin | 0.027 | False | False |
| acute myeloid leukemia and myelodysplastic syndromes related to topoisomerase type 2 inhibitor | 15165 | Midostaurin | 0.027 | False | False |
| acute myeloblastic leukemia without maturation | 5224 | Midostaurin | 0.027 | False | False |
| acute myeloid leukemia with minimal differentiation | 5223 | Midostaurin | 0.027 | False | False |
| acute myeloblastic leukemia with maturation | 20320 | Midostaurin | 0.027 | False | False |
| acute myeloid leukemia with t(8;16)(p11;p13) translocation | 18256 | Midostaurin | 0.027 | False | False |
| therapy related acute myeloid leukemia and myelodysplastic syndrome | 19457 | Midostaurin | 0.027 | False | False |
| inherited acute myeloid leukemia | 17893 | Midostaurin | 0.026 | False | False |
| unclassified acute myeloid leukemia | 15667 | Midostaurin | 0.026 | False | False |
| myeloid leukemia | 18874_4643 | Midostaurin | 0.024 | False | False |

## nsiad: NSIAD

| raw id | internal idx | name | degree | phenotype edges |
|---|---|---|---|---|
| 10356 | 437.0 | nephrogenic syndrome of inappropriate antidiuresis | 34 | 11 |
| 6802 | 14321.0 | inappropriate ADH syndrome | 23 | 0 |

Causal/associated genes of eval and held-out nodes (augmented graph):

- nephrogenic syndrome of inappropriate antidiuresis (10356): 1 genes: AVPR2
- inappropriate ADH syndrome (6802): 0 genes: 

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Tolvaptan | DB06212 | 10356 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| polycystic kidney disease | 33281_8263_33004_13131_9889_4691_20642 | Tolvaptan | 0.025 | False | False |
| inappropriate ADH syndrome | 6802 | Tolvaptan | 0.000 | False | True |

## pku: phenylketonuria

| raw id | internal idx | name | degree | phenotype edges |
|---|---|---|---|---|
| 100186 | 123.0 | GTP cyclohydrolase I deficiency with hyperphenylalaninemia | 30 | 26 |
| 44304 | 11901.0 | hyperphenylalaninemia due to DNAJC12 deficiency | 18 | 13 |
| 9863 | 16953.0 | BH4-deficient hyperphenylalaninemia A | 47 | 45 |
| 9861_19259_16366_19258 | 16951.0 | phenylketonuria | 110 | 75 |
| 17306 | 4948.0 | disorder of phenylalanine metabolism | 26 | 0 |
| 16543 | 4263.0 | hyperphenylalaninemia due to tetrahydrobiopterin deficiency | 26 | 18 |
| 17389 | 5027.0 | tetrahydrobiopterin-responsive hyperphenylalaninemia/phenylketonuria | 2 | 0 |
| 19335 | 6581.0 | mild hyperphenylalaninemia | 1 | 0 |

Causal/associated genes of eval and held-out nodes (augmented graph):

- phenylketonuria (9861_19259_16366_19258): 2 genes: PAH, QDPR
- disorder of phenylalanine metabolism (17306): 2 genes: PAH, QDPR

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Sapropterin | DB00360 | 9861_19259_16366_19258 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| disorder of phenylalanine metabolism | 17306 | Sapropterin | 0.086 | True | True |

## mh: malignant hyperthermia

| raw id | internal idx | name | degree | phenotype edges |
|---|---|---|---|---|
| 11163_7783_7939_10893_7940_11164 | 1040.0 | malignant hyperthermia, susceptibility to | 20 | 17 |
| 9012 | 16203.0 | multiple pterygium-malignant hyperthermia syndrome | 62 | 56 |
| 18493 | 5912.0 | malignant hyperthermia of anesthesia | 36 | 22 |
| 18752 | 6125.0 | exercise-induced malignant hyperthermia | 42 | 30 |

Causal/associated genes of eval and held-out nodes (augmented graph):

- malignant hyperthermia of anesthesia (18493): 2 genes: CACNA1S, RYR1
- exercise-induced malignant hyperthermia (18752): 2 genes: CACNA1S, RYR1

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Dantrolene | DB01219 | 18493, 18752 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| neuroleptic malignant syndrome | 19790 | Dantrolene | 0.203 | False | False |
| Becker muscular dystrophy | 10311 | Dantrolene | 0.082 | False | False |
| Duchenne muscular dystrophy | 10679 | Dantrolene | 0.047 | False | False |
| muscular dystrophy | 11486_12279_19950_20121_23595_18281 | Dantrolene | 0.019 | False | False |

## tyr1: tyrosinemia type 1

| raw id | internal idx | name | degree | phenotype edges |
|---|---|---|---|---|
| 10162_10160_10161_4741 | 311.0 | tyrosinemia | 77 | 65 |
| 18083 | 5587.0 | transient tyrosinemia of the newborn | 1 | 0 |

Causal/associated genes of eval and held-out nodes (augmented graph):

- tyrosinemia (10162_10160_10161_4741): 3 genes: FAH, HPD, TAT

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Nitisinone | DB00348 | 10162_10160_10161_4741 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|

# Step 2: blindness and prototype audit (real disease_eval split)


## cel: holdout [1502.0, 3577.0, 3582.0, 13744.0] = ['chronic eosinophilic leukemia', 'chronic eosinophilic leukemia, not otherwise specified', 'hypereosinophilic syndrome', 'idiopathic hypereosinophilic syndrome']
- blind: **True** (0 held-out drug-disease rows in train)
- eval chronic eosinophilic leukemia (3577.0): profile size 3, best cos 0.1333  **UNRELIABLE (best cos < 0.2)**
  - lookalike 'monocytic leukemia' (idx 15262.0) cos=0.1333
  - lookalike 'myeloid leukemia' (idx 6226.0) cos=0.1184
  - lookalike 'congestive heart failure' (idx 12648.0) cos=0.1077
- eval chronic eosinophilic leukemia, not otherwise specified (13744.0): profile size 4, best cos 0.1732  **UNRELIABLE (best cos < 0.2)**
  - lookalike 'monocytic leukemia' (idx 15262.0) cos=0.1732
  - lookalike 'myeloid leukemia' (idx 6226.0) cos=0.1025
  - lookalike 'congestive heart failure' (idx 12648.0) cos=0.0933

## sm: holdout [3478.0, 3479.0, 4301.0, 7351.0, 7352.0, 15301.0] = ['obsolete classic mast cell leukemia', 'mastocytosis', 'mast cell leukemia', 'aggressive systemic mastocytosis', 'aleukemic mast cell leukemia', 'systemic mastocytosis']
- blind: **True** (0 held-out drug-disease rows in train)
- eval systemic mastocytosis (4301.0): profile size 8, best cos 0.1768  **UNRELIABLE (best cos < 0.2)**
  - lookalike 'gastroparesis (disease)' (idx 14288.0) cos=0.1768
  - lookalike 'acute myeloid leukemia with abnormal bone marrow eosinophils inv(16)(p13q22) or t(16;16)(p13;q22)' (idx 7334.0) cos=0.1443
  - lookalike 'seminoma' (idx 9950.0) cos=0.0913

## nsiad: holdout [437.0, 14321.0] = ['inappropriate ADH syndrome', 'nephrogenic syndrome of inappropriate antidiuresis']
- blind: **True** (0 held-out drug-disease rows in train)
- eval nephrogenic syndrome of inappropriate antidiuresis (437.0): profile size 13, best cos 0.1754  **UNRELIABLE (best cos < 0.2)**
  - lookalike 'Bartter disease' (idx 1464.0) cos=0.1754
  - lookalike 'cholera' (idx 3641.0) cos=0.1545
  - lookalike 'pituitary deficiency due to empty sella turcica syndrome' (idx 6803.0) cos=0.124

## pku: holdout [4948.0, 16951.0] = ['disorder of phenylalanine metabolism', 'phenylketonuria']
- blind: **True** (0 held-out drug-disease rows in train)
- eval phenylketonuria (16951.0): profile size 90, best cos 0.1784  **UNRELIABLE (best cos < 0.2)**
  - lookalike 'argininosuccinic aciduria' (idx 16029.0) cos=0.1784
  - lookalike 'hyperargininemia' (idx 16028.0) cos=0.1677
  - lookalike 'Lennox-Gastaut syndrome' (idx 4253.0) cos=0.1675

## mh: holdout [5912.0, 6125.0] = ['exercise-induced malignant hyperthermia', 'malignant hyperthermia of anesthesia']
- blind: **True** (0 held-out drug-disease rows in train)
- eval malignant hyperthermia of anesthesia (5912.0): profile size 27, best cos 0.134  **UNRELIABLE (best cos < 0.2)**
  - lookalike 'tetanus' (idx 13107.0) cos=0.134
  - lookalike 'anxiety' (idx 1512.0) cos=0.1111
  - lookalike 'cholera' (idx 3641.0) cos=0.1072
- eval exercise-induced malignant hyperthermia (6125.0): profile size 33, best cos 0.2438
  - lookalike 'hyperammonemia due to N-acetylglutamate synthase deficiency' (idx 16530.0) cos=0.2438
  - lookalike 'cholera' (idx 3641.0) cos=0.2263
  - lookalike 'citrullinemia' (idx 16181.0) cos=0.1797

## tyr1: holdout [311.0] = ['tyrosinemia']
- blind: **True** (0 held-out drug-disease rows in train)
- eval tyrosinemia (311.0): profile size 74, best cos 0.2289
  - lookalike 'galactosemia' (idx 9996.0) cos=0.2289
  - lookalike 'argininosuccinic aciduria' (idx 16029.0) cos=0.227
  - lookalike 'scrub typhus' (idx 6601.0) cos=0.2136

# Problems
none
