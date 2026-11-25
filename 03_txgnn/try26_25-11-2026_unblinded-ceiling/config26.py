# Single source of ids/paths for try26's three jobs. Nothing is re-derived downstream.
#
# UNBLINDED = the (disease, indication, drug) edge is in the TRAINING graph. No retraining
# is needed to get that: each try23/try24 run held out only its own disease, so every OTHER
# disease's indication edges are already in that run's train split. Scoring disease D with a
# model that did not hold out D is therefore an unblinded evaluation, available by inference
# on the existing checkpoints (the same trick try19 used for its 150 background diseases).
import os

DATA_DIR = '/work/hdd/bisa/sshah21/primekg_aug'
V3 = '/work/hdd/bisa/sshah21/results/txgnn_v3_augmented_symptoms'     # try23 checkpoints
V4 = '/work/hdd/bisa/sshah21/results/txgnn_v4_new_disease_set'        # try24 checkpoints
OUT = '/work/hdd/bisa/sshah21/results/txgnn_v5_unblinded'
POOL_DIR = '/work/hdd/bisa/sshah21/results/equalized_pool'
TRY25 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'try25_23-11-2026_equalized-pool', 'results_pool_v1_ensemble.csv')
EDGE_AUDIT = os.path.join(OUT, 'edge_audit.csv')
NODE_IDS = os.path.join(OUT, 'node_ids.csv')   # x_idx -> raw PrimeKG id (prediction-dict key)
SEEDS = [1, 2, 3]

# model name -> (results root, disease_eval_idx it was TRAINED with)
MODELS = {
    'fabry':   (V3, [543.0, 6521.0]),
    'gaucher': (V3, [16430.0]),
    'cf':      (V3, [16250.0]),
    'ttr':     (V3, [6669.0, 6361.0, 6665.0, 6008.0, 4803.0, 5528.0]),
    'cel':     (V4, [1502.0, 3577.0, 3582.0, 13744.0]),
    'sm':      (V4, [3478.0, 3479.0, 4301.0, 7351.0, 7352.0, 15301.0]),
    'nsiad':   (V4, [437.0, 14321.0]),
    'pku':     (V4, [4948.0, 16951.0]),
    'mh':      (V4, [5912.0, 6125.0]),
    'tyr1':    (V4, [311.0]),
}
ORDER = ['fabry', 'gaucher', 'cf', 'ttr', 'cel', 'sm', 'nsiad', 'pku', 'mh', 'tyr1']

# target disease node (x_idx) -> (owning run, node name, {DrugBank id: drug name})
# Same nodes and reference drugs try25 reported, so the two tables line up row for row.
TARGETS = {
    543.0:   ('fabry',   'Fabry disease',                    {'DB05018': 'migalastat'}),
    16430.0: ('gaucher', 'Gaucher disease',                  {'DB00419': 'miglustat',
                                                              'DB09039': 'eliglustat'}),
    16250.0: ('cf',      'cystic fibrosis',                  {'DB08820': 'ivacaftor',
                                                              'DB09280': 'lumacaftor',
                                                              'DB11712': 'tezacaftor',
                                                              'DB15444': 'elexacaftor'}),
    6669.0:  ('ttr',     'ATTRV122I amyloidosis',            {'DB11644': 'tafamidis'}),
    5528.0:  ('ttr',     'wild type ATTR amyloidosis',       {'DB11644': 'tafamidis'}),
    # 5528 and 4803 carry NO tafamidis indication edge anywhere in PrimeKG (checked on
    # kg_directed.csv: tafamidis's only indication edges go to 6669 ATTRV122I, 6361
    # amyloidosis, 6665 AL amyloidosis, 6008 AH amyloidosis). They are kept here so job A
    # records that absence, but there is no edge to put in train, so they can never be
    # evaluated unblinded -- and try23's blinded ranks at them were ranks with no
    # ground-truth edge behind them. Job B skips them.
    4803.0:  ('ttr',     'hereditary ATTR amyloidosis',      {'DB11644': 'tafamidis'}),
    3577.0:  ('cel',     'chronic eosinophilic leukemia',    {'DB00619': 'imatinib'}),
    13744.0: ('cel',     'chronic eosinophilic leukemia, not otherwise specified', {'DB00619': 'imatinib'}),
    4301.0:  ('sm',      'systemic mastocytosis',            {'DB06595': 'midostaurin'}),
    437.0:   ('nsiad',   'nephrogenic syndrome of inappropriate antidiuresis',{'DB06212': 'tolvaptan'}),
    16951.0: ('pku',     'phenylketonuria',                  {'DB00360': 'sapropterin'}),
    5912.0:  ('mh',      'malignant hyperthermia of anesthesia', {'DB01219': 'dantrolene'}),
    6125.0:  ('mh',      'exercise-induced malignant hyperthermia', {'DB01219': 'dantrolene'}),
    311.0:   ('tyr1',    'tyrosinemia',                      {'DB00348': 'nitisinone'}),
}

# ---- REPORTING RULE, fixed before any unblinded number was looked at -------------------
# A (model, target node, drug) cell counts as unblinded only if job A finds that exact
# indication edge in that split's TRAIN file. 'valid' and 'test' do not count: valid/test
# edges are held out of the message-passing graph, so the model never saw them.
# Several models are usually eligible for one target. The headline unblinded rank is the
# MEDIAN across eligible models of the seed-ensembled rank; min/max/n_models are reported
# beside it. Picking the best-looking model would be selecting on the outcome.
HEADLINE = 'median'
