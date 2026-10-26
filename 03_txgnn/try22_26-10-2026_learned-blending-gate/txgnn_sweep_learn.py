import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from txgnn import TxData, TxGNN, TxEval

DATA_DIR = '/work/hdd/bisa/sshah21/primekg'
# Same checkpoint-selection criterion as try11 (installed library patch: indication
# AUROC), so results live alongside fabry/gaucher under the v2 dir.
RESULTS_DIR = '/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection'
DD_RELATIONS = 'indication|contraindication|off-label'
SEEDS = [1, 2, 3]

# 120 diseases sampled by sample_diseases.py (6 log-degree bins x 20), held out
# TOGETHER: one model per seed scores all of them. Thesis targets excluded.
# Same 120 diseases as try16 (read from try16's folder) -> same cached split, paired comparison.
SAMPLE = pd.read_csv('/projects/bisa/sshah21/library/03_txgnn/try16_26-10-2026_connectivity-vs-rank/sample.csv')


class SplitTag(list):
    # TxData names the cached split folder 'disease_eval_' + str(disease_eval_idx)
    # (TxData.py:72). str() of 120 floats is >1,000 chars, past the 255-char filename
    # limit, so os.mkdir fails. A list subclass with a short str() keeps the folder
    # name sane; disease_eval_fold only does isinstance(list) + np.array(), and the
    # value is never pickled into the model config (TxGNN.py:91), so this is safe
    # without patching the library.
    def __str__(self):
        return '[connectivity_sweep_120]'
    __repr__ = __str__


HOLDOUT = SplitTag(float(i) for i in SAMPLE.idx)
# 2x2 arm: agg_measure='learn' (learned blending gate; needs the try22 library patch
# that registers W_gate as nn.ModuleDict) with sim_measure='all_nodes_profile'.
NAME = 'connectivity_sweep_learn'


def set_seed(seed):
    # TxGNN.py hardcodes torch.manual_seed(0) at import time -- override it here so
    # each seed actually gets an independently initialized/trained model, not just a
    # different data split with identical weight init.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_blind(df_train, disease_idxs, name):
    rel = df_train.relation.str.contains(DD_RELATIONS, case=False, regex=True)
    leaked = df_train[rel & (((df_train.x_type == 'disease') & df_train.x_idx.isin(disease_idxs)) |
                              ((df_train.y_type == 'disease') & df_train.y_idx.isin(disease_idxs)))]
    if len(leaked):
        print(f"LEAKAGE: {len(leaked)} drug<->{name} edges are in TRAIN -- not blind:")
        cols = [c for c in ['x_id', 'x_type', 'relation', 'y_id', 'y_type'] if c in leaked.columns]
        print(leaked[cols].drop_duplicates().to_string())
        return False
    print(f"OK: none of {name}'s held-out diseases {disease_idxs} has drug edges in train -- model is blind.")
    return True


def run_disease_eval(name, holdout, seed, result_dir):
    set_seed(seed)
    ckpt = os.path.join(result_dir, 'ckpt')
    os.makedirs(result_dir, exist_ok=True)

    d = TxData(data_folder_path=DATA_DIR)
    d.prepare_split(disease_eval_idx=holdout, seed=seed)

    print(f"=== [{name} seed={seed}] Blindness check ===")
    if not check_blind(d.df_train, holdout, name):
        raise RuntimeError(f'Refusing to report {name} (seed {seed}) results: leakage detected.')

    set_seed(seed)  # re-seed right before model init, since prepare_split() may have burned RNG calls
    model = TxGNN(data=d, weight_bias_track=False, proj_name='TxGNN', exp_name=f'{name}_s{seed}', device='cuda:0')
    if os.path.exists(os.path.join(ckpt, 'model.pt')):
        print(f'[{name} seed={seed}] Loading existing checkpoint (skipping retrain)...')
        model.load_pretrained(ckpt)
    else:
        model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3,
                                attention=True, sim_measure='all_nodes_profile', agg_measure='learn')
        # the gate must actually be trainable (try22 patch) -- fail fast otherwise
        gate = [n for n, _ in model.model.named_parameters() if 'W_gate' in n]
        assert gate, 'W_gate not registered: try22 library patch missing, gate would be random/untrained'
        print('learned-gate parameters registered:', gate)
        # pretrain() skipped with attention=True -- DGL block-sampling crash on
        # same-type relations, see try09 README.
        model.finetune(n_epoch=500, learning_rate=5e-4)
        model.save_model(ckpt)

    # one call for all held-out diseases: the expensive preamble in
    # disease_centric_evaluation (re-reading kg.csv etc.) then runs once, not 120x.
    # simulate_random only feeds the library's Recall_Random metric, which we don't use.
    ev = TxEval(model=model)
    save = os.path.join(result_dir, 'eval_all.pkl')
    out = ev.eval_disease_centric(disease_idxs=list(holdout), relation='indication', save_result=True,
                                  return_raw=True, save_name=save, simulate_random=False)
    print(f'[{name} seed={seed}] eval_disease_centric over {len(holdout)} diseases saved to {save}')
    return out


for seed in SEEDS:
    run_disease_eval(NAME, HOLDOUT, seed, os.path.join(RESULTS_DIR, NAME, f'seed_{seed}'))
print('done; run analyze.py on CPU for the connectivity-vs-rank analysis')
