# try24 fixed configuration, shared by every step of the chain. Do not edit while jobs are queued.
import os

TRY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/work/hdd/bisa/sshah21/primekg_aug'
RESULTS_DIR = '/work/hdd/bisa/sshah21/results/txgnn_v4_new_disease_set'
RESOLVED = os.path.join(RESULTS_DIR, 'resolved.json')   # written by job A, read by B and C
SIM = 'all_nodes_profile_more'   # fixed on try23 results, before any new-disease run
SEEDS = [1, 2, 3]
UNRELIABLE_COS = 0.2             # try20 / TxGNN paper threshold
NEAR_DUP_COS = 0.5               # answer-carrying disease this similar to the query must be held out

# Order is fixed: SLURM_ARRAY_TASK_ID i runs ORDER[i]. Diseases that job A marks not
# evaluable are skipped by their array task.
ORDER = ['cel', 'sm', 'nsiad', 'pku', 'mh', 'tyr1']

# All ids are RAW PrimeKG node ids (node.csv node_id). Job A maps them to TxGNN's
# per-type disease index (kg_directed.csv x_idx/y_idx) and checks the names.
#   name_regex : every disease whose name matches is printed in the resolution report
#   eval       : nodes we score drugs against (raw id -> expected name)
#   holdout_extra : other nodes holding the answer (parents, subtypes, near-duplicates)
#   ref        : reference drug name(s); first is primary. Resolved to DrugBank ids by name.
DISEASES = {
    'cel': dict(
        label='chronic eosinophilic leukemia',
        name_regex=r'eosinophilic leuk|hypereosinophil',
        eval={'15687': 'chronic eosinophilic leukemia',
              '6147': 'chronic eosinophilic leukemia, not otherwise specified'},
        holdout_extra={},
        ref=['Imatinib']),
    'sm': dict(
        label='systemic mastocytosis',
        name_regex=r'mastocytosis|mast cell leuk',
        eval={'16586': 'systemic mastocytosis'},
        holdout_extra={},
        ref=['Midostaurin', 'Avapritinib']),
    'nsiad': dict(
        label='NSIAD',
        name_regex=r'antidiure|antidiuretic|ADH syndrome',
        eval={'10356': 'nephrogenic syndrome of inappropriate antidiuresis'},
        holdout_extra={},
        ref=['Tolvaptan']),
    'pku': dict(
        label='phenylketonuria',
        name_regex=r'phenylketon|hyperphenylalanin|phenylalanine metab',
        eval={'9861_19259_16366_19258': 'phenylketonuria'},
        holdout_extra={},
        ref=['Sapropterin']),
    'mh': dict(
        label='malignant hyperthermia',
        name_regex=r'malignant hyperthermia',
        eval={'18493': 'malignant hyperthermia of anesthesia'},
        holdout_extra={},
        ref=['Dantrolene']),
    'tyr1': dict(
        label='tyrosinemia type 1',
        name_regex=r'tyrosin(a)?emia',
        eval={'10162_10160_10161_4741': 'tyrosinemia'},
        holdout_extra={},
        ref=['Nitisinone']),
}
