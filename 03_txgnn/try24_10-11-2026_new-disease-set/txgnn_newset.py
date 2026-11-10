# Job B (GPU array task). Same model/training as try23's txgnn_aug_symptoms.py (copied
# config, unchanged): augmented PrimeKG, sim_measure='all_nodes_profile_more', 3 seeds.
# Disease = config.ORDER[SLURM_ARRAY_TASK_ID]; holdout/eval lists come from job A's
# resolved.json, so nothing is re-derived here.
import json
import os
import random
import sys

import numpy as np
import torch
from txgnn import TxData, TxGNN, TxEval

from config import DATA_DIR, ORDER, RESOLVED, RESULTS_DIR, SEEDS, SIM

DD_RELATIONS = 'indication|contraindication|off-label'


def set_seed(seed):
    # TxGNN.py hardcodes torch.manual_seed(0) at import -- override per seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_blind(df_train, disease_idxs, name):
    rel = df_train.relation.str.contains(DD_RELATIONS, case=False, regex=True)
    leaked = df_train[rel & (((df_train.x_type == 'disease') & df_train.x_idx.isin(disease_idxs)) |
                              ((df_train.y_type == 'disease') & df_train.y_idx.isin(disease_idxs)))]
    if len(leaked):
        print(f'LEAKAGE: {len(leaked)} drug<->{name} edges are in TRAIN -- not blind')
        return False
    print(f"OK: none of {name}'s held-out diseases {disease_idxs} has drug edges in train -- model is blind.")
    return True


def run(name, info, seed):
    result_dir = os.path.join(RESULTS_DIR, name, f'seed_{seed}')
    ckpt = os.path.join(result_dir, 'ckpt')
    os.makedirs(result_dir, exist_ok=True)
    set_seed(seed)
    d = TxData(data_folder_path=DATA_DIR)
    d.prepare_split(disease_eval_idx=info['holdout'], seed=seed)
    blind = check_blind(d.df_train, info['holdout'], name)
    json.dump({'blind_check_passed': blind}, open(os.path.join(result_dir, 'blind.json'), 'w'))
    if not blind:
        raise RuntimeError(f'Refusing to run {name} seed {seed}: leakage detected.')
    set_seed(seed)  # re-seed right before model init (prepare_split may burn RNG calls)
    model = TxGNN(data=d, weight_bias_track=False, proj_name='TxGNN', exp_name=f'{name}_s{seed}', device='cuda:0')
    if os.path.exists(os.path.join(ckpt, 'model.pt')):
        print(f'[{name} seed={seed}] loading existing checkpoint')
        model.load_pretrained(ckpt)
    else:
        model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3,
                               attention=True, sim_measure=SIM, agg_measure='rarity')
        # pretrain() skipped with attention=True -- DGL block-sampling crash, see try09
        model.finetune(n_epoch=500, learning_rate=5e-4)
        model.save_model(ckpt)
    ev = TxEval(model=model)
    for q in info['eval']:
        save = os.path.join(result_dir, f'eval_{int(float(q))}.pkl')
        ev.eval_disease_centric(disease_idxs=[float(q)], relation='indication',
                                save_result=True, return_raw=True, save_name=save)
        print(f'[{name} seed={seed}] saved {save}', flush=True)


if __name__ == '__main__':
    name = ORDER[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    info = json.load(open(RESOLVED))[name]
    if not info['ok']:
        print(f'{name}: marked not evaluable / failed by job A -- skipping (see resolution_report.md)')
        sys.exit(0)
    for seed in SEEDS:
        run(name, info, seed)
    print(f'{name}: all seeds done')
