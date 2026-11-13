import json
import os
import pickle
import random

import numpy as np
import torch
from txgnn import TxData, TxGNN, TxEval

DATA_DIR = '/work/hdd/bisa/sshah21/primekg'
RESULTS_DIR = '/work/hdd/bisa/sshah21/results/txgnn'
DD_RELATIONS = 'indication|contraindication|off-label'
SEEDS = [1, 2, 3]

# PrimeKG's TxGNN-internal, per-node-type index (from kg_directed.csv x_idx/y_idx) --
# NOT the raw global node.csv/kg.csv row index.
DISEASES = {
    # complex_disease's blindness for a given disease is seed-dependent luck, not
    # structural -- seed=42 happened to hold Fabry out, seed=1 leaks it. disease_eval
    # removes a named disease's drug edges from train by construction regardless of
    # seed, so it's used for both here rather than relying on a lucky split.
    'fabry':   {'idx': 543.,   'split': lambda seed: dict(disease_eval_idx=[543.], seed=seed)},
    'gaucher': {'idx': 16430., 'split': lambda seed: dict(disease_eval_idx=[16430.], seed=seed)},
}

# reference drugs we're actually watching the rank of, per disease
REFERENCE_DRUGS = {
    'fabry':   {'DB05018': 'migalastat'},
    'gaucher': {'DB06720': 'velaglucerase alfa', 'DB08876': 'taliglucerase alfa',
                'DB00088': 'alglucerase', 'DB04878': 'voglibose',
                'DB00419': 'miglustat', 'DB09039': 'eliglustat', 'DB00053': 'imiglucerase'},
}


def set_seed(seed):
    # TxGNN.py hardcodes torch.manual_seed(0) at import time -- override it here so
    # each seed actually gets an independently initialized/trained model, not just a
    # different data split with identical weight init.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_blind(df_train, disease_idx, name):
    rel = df_train.relation.str.contains(DD_RELATIONS, case=False, regex=True)
    leaked = df_train[rel & (((df_train.x_type == 'disease') & (df_train.x_idx == disease_idx)) |
                              ((df_train.y_type == 'disease') & (df_train.y_idx == disease_idx)))]
    if len(leaked):
        print(f"LEAKAGE: {len(leaked)} drug<->{name} edges are in TRAIN -- not blind:")
        cols = [c for c in ['x_id', 'x_type', 'relation', 'y_id', 'y_type'] if c in leaked.columns]
        print(leaked[cols].drop_duplicates().to_string())
        return False
    print(f"OK: {name} has no drug edges in the training split -- model is blind to it.")
    return True


def run_disease_eval(name, idx, split_kwargs, seed, result_dir):
    set_seed(seed)
    ckpt = os.path.join(result_dir, 'ckpt')
    os.makedirs(result_dir, exist_ok=True)

    d = TxData(data_folder_path=DATA_DIR)
    d.prepare_split(**split_kwargs)

    print(f"=== [{name} seed={seed}] Blindness check ===")
    if not check_blind(d.df_train, idx, name):
        raise RuntimeError(f'Refusing to report {name} (seed {seed}) results: leakage detected.')

    set_seed(seed)  # re-seed right before model init, since prepare_split() may have burned RNG calls
    model = TxGNN(data=d, weight_bias_track=False, proj_name='TxGNN', exp_name=f'{name}_s{seed}', device='cuda:0')
    if os.path.exists(os.path.join(ckpt, 'model.pt')):
        print(f'[{name} seed={seed}] Loading existing checkpoint (skipping retrain)...')
        model.load_pretrained(ckpt)
    else:
        model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3,
                                attention=True, sim_measure='all_nodes_profile', agg_measure='rarity')
        # pretrain() uses DGL's minibatch/block sampling for the whole-graph warm-start
        # stage. In a block, same-type relations (e.g. anatomy_anatomy) have separate
        # source/destination node stores even though both sides are "anatomy" -- and
        # AttHeteroRGCNLayer (attention=True) only ever writes its computed feature to
        # one side, then crashes reading it from the other (KeyError: 'Wh_<etype>').
        # finetune() doesn't hit this because it runs on the full graph, not blocks.
        # Skip the generic warm-start and go straight to the disease-specific stage.
        model.finetune(n_epoch=500, learning_rate=5e-4)
        model.save_model(ckpt)

    ev = TxEval(model=model)
    out = ev.eval_disease_centric(disease_idxs=[idx], relation='indication',
                                   save_result=True, return_raw=True,
                                   save_name=os.path.join(result_dir, 'eval.pkl'))
    print(f'[{name} seed={seed}] eval_disease_centric result saved to {result_dir}/eval.pkl')
    return out


def rank_of(out, disease_idx, drug_id):
    preds = out['prediction'][str(disease_idx)]
    ranked = sorted(preds.items(), key=lambda x: -x[1])
    for i, (drug, score) in enumerate(ranked):
        if drug == drug_id:
            return i + 1, len(ranked), score
    return None, len(ranked), None


def report_aggregate(name, idx, outs):
    print(f"\n=== {name.upper()} aggregate over seeds {SEEDS} (attention=True) ===")
    for drug_id, drug_name in REFERENCE_DRUGS[name].items():
        ranks, scores = [], []
        for seed, out in zip(SEEDS, outs):
            r, total, s = rank_of(out, idx, drug_id)
            print(f"  seed {seed}: {drug_name} ({drug_id}) rank {r}/{total}, score {s}")
            if r is not None:
                ranks.append(r)
                scores.append(s)
        if ranks:
            print(f"  -> {drug_name}: mean rank {sum(ranks) / len(ranks):.1f}, "
                  f"mean score {sum(scores) / len(scores):.4f}")


only_disease = os.environ.get('DISEASE')
targets = {only_disease: DISEASES[only_disease]} if only_disease else DISEASES

for disease_name, cfg in targets.items():
    outs = []
    for seed in SEEDS:
        result_dir = os.path.join(RESULTS_DIR, disease_name, f'seed_{seed}')
        outs.append(run_disease_eval(disease_name, cfg['idx'], cfg['split'](seed), seed, result_dir))
    report_aggregate(disease_name, cfg['idx'], outs)
    with open(os.path.join(RESULTS_DIR, disease_name, 'all_seeds_raw.pkl'), 'wb') as f:
        pickle.dump(outs, f)
