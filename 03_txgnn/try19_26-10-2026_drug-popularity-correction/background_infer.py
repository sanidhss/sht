# Inference only (no training): reload each thesis run's saved checkpoints and score
# the 150 background diseases, so per-drug popularity can be measured with the SAME
# model that scored the thesis disease. Skips seeds already done or not yet trained;
# safe to resubmit (e.g. after try15 finishes).
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from txgnn import TxData, TxGNN, TxEval

DATA_DIR = '/work/hdd/bisa/sshah21/primekg'
R = '/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection'
HERE = os.path.dirname(os.path.abspath(__file__))
BG = [float(i) for i in pd.read_csv(os.path.join(HERE, 'background.csv')).idx]
RUNS = {  # results subdir -> (holdout it was trained with, its eval disease, saved eval pkl)
    'fabry': ([543.], 543., 'eval.pkl'), 'gaucher': ([16430.], 16430., 'eval.pkl'),
    'cf': ([16250.], 16250., 'eval_16250.pkl'),
    'ttr': ([6669., 6361., 6665., 6008., 4803., 5528.], 6669., 'eval_6669.pkl'),
    'gaucher_no_np': ([16430., 1490.], 16430., 'eval_16430.pkl'),
}
# Node INPUT vectors are random, frozen, stored on the graph (utils.py L1076
# initialize_node_embedding) and NOT saved in the checkpoint. A reload reproduces the
# trained model only if torch's RNG is in the same state when they're drawn -- the
# training script re-seeds immediately before TxGNN(), so we do exactly the same, and
# then VERIFY by re-scoring the run's own eval disease against its saved scores.
TOL = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scores(out):
    return out['prediction']
for name, (holdout, eval_idx, eval_pkl) in RUNS.items():
    bg = [i for i in BG if i not in holdout]
    for seed in [1, 2, 3]:
        sdir = os.path.join(R, name, f'seed_{seed}')
        out_path = os.path.join(sdir, 'background_150.pkl')
        if os.path.exists(out_path):
            print(f'[{name} s{seed}] done already, skipping'); continue
        if not os.path.exists(os.path.join(sdir, 'ckpt', 'model.pt')):
            print(f'[{name} s{seed}] no checkpoint yet, skipping'); continue
        set_seed(seed)
        d = TxData(data_folder_path=DATA_DIR)
        d.prepare_split(disease_eval_idx=holdout, seed=seed)      # cached split, same as training
        set_seed(seed)  # same re-seed point as the training script, right before TxGNN()
        model = TxGNN(data=d, weight_bias_track=False, proj_name='TxGNN', exp_name=f'{name}_bg_s{seed}', device='cuda:0')
        model.load_pretrained(os.path.join(sdir, 'ckpt'))
        out = TxEval(model=model).eval_disease_centric(disease_idxs=[eval_idx] + bg, relation='indication',
                                                       return_raw=True, simulate_random=False)
        # reproduction check against the original run's saved scores for its own disease
        (key, orig), = pickle.load(open(os.path.join(sdir, eval_pkl), 'rb'))['prediction'].items()
        new = out['prediction'].get(key)  # same disease -> same raw-id key format (try10 note)
        diff = max(abs(new[x] - orig[x]) for x in orig) if new is not None else float('inf')
        print(f'[{name} s{seed}] reproduction check: max |score diff| on own disease = {diff:.2e}')
        if diff > TOL:
            raise RuntimeError(f'{name} s{seed}: reloaded model does NOT reproduce saved scores -- background unusable')
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)
        print(f'[{name} s{seed}] background scores saved to {out_path}')
