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

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Imatinib | DB00619 | 15687, 6147 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| hypereosinophilic syndrome | 15691 | Imatinib | 0.246 | False | False |
| idiopathic hypereosinophilic syndrome | 11895 | Imatinib | 0.130 | False | False |
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

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Midostaurin | DB06595 | 16586 |
| Avapritinib | DB15233 | NONE - not evaluable for this disease |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| aleukemic mast cell leukemia | 15561 | Midostaurin | 0.500 | False | False |
| obsolete classic mast cell leukemia | 15560 | Midostaurin | 0.354 | False | False |
| mast cell leukemia | 20334 | Midostaurin | 0.177 | True | False |
| mastocytosis | 7950 | Midostaurin | 0.120 | True | False |
| aggressive systemic mastocytosis | 20333 | Midostaurin | 0.088 | True | False |
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

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Tolvaptan | DB06212 | 10356 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| polycystic kidney disease | 33281_8263_33004_13131_9889_4691_20642 | Tolvaptan | 0.025 | False | False |
| inappropriate ADH syndrome | 6802 | Tolvaptan | 0.000 | False | False |

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

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Sapropterin | DB00360 | 9861_19259_16366_19258 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| disorder of phenylalanine metabolism | 17306 | Sapropterin | 0.086 | True | False |

## mh: malignant hyperthermia

| raw id | internal idx | name | degree | phenotype edges |
|---|---|---|---|---|
| 11163_7783_7939_10893_7940_11164 | 1040.0 | malignant hyperthermia, susceptibility to | 20 | 17 |
| 9012 | 16203.0 | multiple pterygium-malignant hyperthermia syndrome | 62 | 56 |
| 18493 | 5912.0 | malignant hyperthermia of anesthesia | 36 | 22 |
| 18752 | 6125.0 | exercise-induced malignant hyperthermia | 42 | 30 |

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Dantrolene | DB01219 | 18493 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|
| exercise-induced malignant hyperthermia | 18752 | Dantrolene | 0.168 | False | False |
| neuroleptic malignant syndrome | 19790 | Dantrolene | 0.112 | False | False |
| muscular dystrophy | 11486_12279_19950_20121_23595_18281 | Dantrolene | 0.000 | False | False |
| Duchenne muscular dystrophy | 10679 | Dantrolene | 0.000 | False | False |
| Becker muscular dystrophy | 10311 | Dantrolene | 0.000 | False | False |

## tyr1: tyrosinemia type 1

| raw id | internal idx | name | degree | phenotype edges |
|---|---|---|---|---|
| 10162_10160_10161_4741 | 311.0 | tyrosinemia | 77 | 65 |
| 18083 | 5587.0 | transient tyrosinemia of the newborn | 1 | 0 |

| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |
|---|---|---|
| Nitisinone | DB00348 | 10162_10160_10161_4741 |

Other diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):

| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |
|---|---|---|---|---|---|

# Step 2: blindness and prototype audit (real disease_eval split)


## cel: holdout [3577.0, 13744.0] = ['chronic eosinophilic leukemia, not otherwise specified', 'chronic eosinophilic leukemia']
- blind: **True** (0 held-out drug-disease rows in train)
- eval chronic eosinophilic leukemia (3577.0): profile size 3, best cos 0.2462
  - lookalike 'hypereosinophilic syndrome' (idx 3582.0) cos=0.2462  <-- LEAK: carries ['Imatinib']
  - lookalike 'monocytic leukemia' (idx 15262.0) cos=0.1333
  - lookalike 'idiopathic hypereosinophilic syndrome' (idx 1502.0) cos=0.1299  <-- LEAK: carries ['Imatinib']
- eval chronic eosinophilic leukemia, not otherwise specified (13744.0): profile size 4, best cos 0.2132
  - lookalike 'hypereosinophilic syndrome' (idx 3582.0) cos=0.2132  <-- LEAK: carries ['Imatinib']
  - lookalike 'monocytic leukemia' (idx 15262.0) cos=0.1732
  - lookalike 'idiopathic hypereosinophilic syndrome' (idx 1502.0) cos=0.1125  <-- LEAK: carries ['Imatinib']

## sm: holdout [4301.0] = ['systemic mastocytosis']
- blind: **True** (0 held-out drug-disease rows in train)
- eval systemic mastocytosis (4301.0): profile size 8, best cos 0.5
  - lookalike 'aleukemic mast cell leukemia' (idx 3479.0) cos=0.5  <-- LEAK: carries ['Midostaurin']
  - lookalike 'gastroparesis (disease)' (idx 14288.0) cos=0.1768
  - lookalike 'mast cell leukemia' (idx 7352.0) cos=0.1768  <-- LEAK: carries ['Midostaurin']

## nsiad: holdout [437.0] = ['nephrogenic syndrome of inappropriate antidiuresis']
- blind: **True** (0 held-out drug-disease rows in train)
- eval nephrogenic syndrome of inappropriate antidiuresis (437.0): profile size 13, best cos 0.1754  **UNRELIABLE (best cos < 0.2)**
  - lookalike 'Bartter disease' (idx 1464.0) cos=0.1754
  - lookalike 'cholera' (idx 3641.0) cos=0.1545
  - lookalike 'pituitary deficiency due to empty sella turcica syndrome' (idx 6803.0) cos=0.124

## pku: holdout [16951.0] = ['phenylketonuria']
- blind: **True** (0 held-out drug-disease rows in train)
- eval phenylketonuria (16951.0): profile size 90, best cos 0.1784  **UNRELIABLE (best cos < 0.2)**
  - lookalike 'argininosuccinic aciduria' (idx 16029.0) cos=0.1784
  - lookalike 'hyperargininemia' (idx 16028.0) cos=0.1677
  - lookalike 'Lennox-Gastaut syndrome' (idx 4253.0) cos=0.1675

## mh: holdout [5912.0] = ['malignant hyperthermia of anesthesia']
- blind: **True** (0 held-out drug-disease rows in train)
- eval malignant hyperthermia of anesthesia (5912.0): profile size 27, best cos 0.1675  **UNRELIABLE (best cos < 0.2)**
  - lookalike 'exercise-induced malignant hyperthermia' (idx 6125.0) cos=0.1675  <-- LEAK: carries ['Dantrolene']
  - lookalike 'tetanus' (idx 13107.0) cos=0.134
  - lookalike 'anxiety' (idx 1512.0) cos=0.1111

## tyr1: holdout [311.0] = ['tyrosinemia']
- blind: **True** (0 held-out drug-disease rows in train)
- eval tyrosinemia (311.0): profile size 74, best cos 0.2289
  - lookalike 'galactosemia' (idx 9996.0) cos=0.2289
  - lookalike 'argininosuccinic aciduria' (idx 16029.0) cos=0.227
  - lookalike 'scrub typhus' (idx 6601.0) cos=0.2136

# Problems
- sm: 'aleukemic mast cell leukemia' (15561) carries Midostaurin, cos=0.500, MONDO neighbour=False, but is NOT held out
- sm: 'mast cell leukemia' (20334) carries Midostaurin, cos=0.177, MONDO neighbour=True, but is NOT held out
- sm: 'mastocytosis' (7950) carries Midostaurin, cos=0.120, MONDO neighbour=True, but is NOT held out
- sm: 'aggressive systemic mastocytosis' (20333) carries Midostaurin, cos=0.088, MONDO neighbour=True, but is NOT held out
- pku: 'disorder of phenylalanine metabolism' (17306) carries Sapropterin, cos=0.086, MONDO neighbour=True, but is NOT held out
- cel: prototype 'hypereosinophilic syndrome' carries ['DB00619']
- cel: prototype 'idiopathic hypereosinophilic syndrome' carries ['DB00619']
- cel: prototype 'hypereosinophilic syndrome' carries ['DB00619']
- cel: prototype 'idiopathic hypereosinophilic syndrome' carries ['DB00619']
- sm: prototype 'aleukemic mast cell leukemia' carries ['DB06595']
- sm: prototype 'mast cell leukemia' carries ['DB06595']
- mh: prototype 'exercise-induced malignant hyperthermia' carries ['DB01219']
