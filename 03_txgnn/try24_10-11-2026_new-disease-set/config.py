# try24 fixed configuration, shared by every step of the chain. Do not edit while jobs are queued.
import os

TRY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/work/hdd/bisa/sshah21/primekg_aug'
# T24_RESULTS_DIR / T24_TAG are set only by the A40 duplicate (B_train_a40.sb); defaults unchanged
RESULTS_DIR = os.environ.get('T24_RESULTS_DIR', '/work/hdd/bisa/sshah21/results/txgnn_v4_new_disease_set')
TAG = os.environ.get('T24_TAG', '')
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
        # both HES nodes carry imatinib and have CEL's exact causal genes (FIP1L1, PDGFRA):
        # FIP1L1-PDGFRA "HES" is the same entity (WHO: myeloid neoplasm with PDGFRA
        # rearrangement). First audit: both were CEL's lookalikes and carried imatinib.
        holdout_extra={'15691': 'hypereosinophilic syndrome',
                       '11895': 'idiopathic hypereosinophilic syndrome'},
        ref=['Imatinib']),
    'sm': dict(
        label='systemic mastocytosis',
        name_regex=r'mastocytosis|mast cell leuk',
        eval={'16586': 'systemic mastocytosis'},
        # parent + subtypes carrying midostaurin (mast cell leukemia is an advanced-SM subtype)
        holdout_extra={'7950': 'mastocytosis', '20333': 'aggressive systemic mastocytosis',
                       '20334': 'mast cell leukemia', '15561': 'aleukemic mast cell leukemia',
                       '15560': 'obsolete classic mast cell leukemia'},
        ref=['Midostaurin', 'Avapritinib']),
    'nsiad': dict(
        label='NSIAD',
        name_regex=r'antidiure|antidiuretic|ADH syndrome',
        eval={'10356': 'nephrogenic syndrome of inappropriate antidiuresis'},
        # name-level near-relative carrying tolvaptan (cos 0, not a MONDO neighbour): held out to be safe
        holdout_extra={'6802': 'inappropriate ADH syndrome'},
        ref=['Tolvaptan']),
    'pku': dict(
        label='phenylketonuria',
        name_regex=r'phenylketon|hyperphenylalanin|phenylalanine metab',
        eval={'9861_19259_16366_19258': 'phenylketonuria'},
        holdout_extra={'17306': 'disorder of phenylalanine metabolism'},   # parent, carries sapropterin
        ref=['Sapropterin']),
    'mh': dict(
        label='malignant hyperthermia',
        name_regex=r'malignant hyperthermia',
        # two MH nodes both carry dantrolene; neither is "the" MH node, so both are held out
        # and both are scored (no silent pick). 18493 is reported as primary.
        eval={'18493': 'malignant hyperthermia of anesthesia',
              '18752': 'exercise-induced malignant hyperthermia'},
        holdout_extra={},
        ref=['Dantrolene']),
    'tyr1': dict(
        label='tyrosinemia type 1',
        name_regex=r'tyrosin(a)?emia',
        eval={'10162_10160_10161_4741': 'tyrosinemia'},
        holdout_extra={},
        ref=['Nitisinone']),
}
