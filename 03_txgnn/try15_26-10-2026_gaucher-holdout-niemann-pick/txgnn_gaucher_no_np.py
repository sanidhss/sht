import os
import pickle
import random

import numpy as np
import torch
from txgnn import TxData, TxGNN, TxEval

DATA_DIR = '/work/hdd/bisa/sshah21/primekg'
# Same checkpoint-selection criterion as try11 (installed library patch: indication
# AUROC), so results live alongside fabry/gaucher under the v2 dir.
RESULTS_DIR = '/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection'
DD_RELATIONS = 'indication|contraindication|off-label'
SEEDS = [1, 2, 3]

# TxGNN-internal per-node-type disease index (kg_directed.csv x_idx/y_idx).
# holdout: every disease whose drug edges are removed from train (disease_eval takes a list).
# eval:    the nodes we score drugs against (one eval_disease_centric call each).
DISEASES = {
    # Same as try11's Gaucher run except Niemann-Pick disease (idx 1490) is also held
    # out. Its only drug edge in all of PrimeKG is miglustat->Niemann-Pick indication,
    # which try14 found sitting 2 hops from Gaucher (shared neighbours incl.
    # sphingolipidosis). So this removes exactly one training edge: does miglustat's
    # ensembled rank (17/7,957) survive without its sibling indication?
    # Results dir name differs from 'gaucher' on purpose -- the script skips training
    # if a checkpoint already exists, and try11's gaucher checkpoints must not be reused.
    'gaucher_no_np': {'holdout': [16430., 1490.], 'eval': {16430.: 'Gaucher disease'}},
}

# reference drugs we're watching the rank of (same 7 as try11; the 3 ERTs are
# isolated in train and will be exact 0.0 -- try12)
REFERENCE_DRUGS = {
    'gaucher_no_np': {'DB00419': 'miglustat', 'DB09039': 'eliglustat',
                      'DB06720': 'velaglucerase alfa', 'DB04878': 'voglibose',
                      'DB00053': 'imiglucerase', 'DB00088': 'alglucerase',
                      'DB08876': 'taliglucerase alfa'},
}


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


def run_disease_eval(name, cfg, seed, result_dir):
    set_seed(seed)
    ckpt = os.path.join(result_dir, 'ckpt')
    os.makedirs(result_dir, exist_ok=True)

    d = TxData(data_folder_path=DATA_DIR)
    d.prepare_split(disease_eval_idx=cfg['holdout'], seed=seed)

    print(f"=== [{name} seed={seed}] Blindness check ===")
    if not check_blind(d.df_train, cfg['holdout'], name):
        raise RuntimeError(f'Refusing to report {name} (seed {seed}) results: leakage detected.')

    set_seed(seed)  # re-seed right before model init, since prepare_split() may have burned RNG calls
    model = TxGNN(data=d, weight_bias_track=False, proj_name='TxGNN', exp_name=f'{name}_s{seed}', device='cuda:0')
    if os.path.exists(os.path.join(ckpt, 'model.pt')):
        print(f'[{name} seed={seed}] Loading existing checkpoint (skipping retrain)...')
        model.load_pretrained(ckpt)
    else:
        model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3,
                                attention=True, sim_measure='all_nodes_profile', agg_measure='rarity')
        # pretrain() skipped with attention=True -- DGL block-sampling crash on
        # same-type relations, see try09 README.
        model.finetune(n_epoch=500, learning_rate=5e-4)
        model.save_model(ckpt)

    ev = TxEval(model=model)
    outs = {}
    for idx in cfg['eval']:
        save = os.path.join(result_dir, f'eval_{int(idx)}.pkl')
        outs[idx] = ev.eval_disease_centric(disease_idxs=[idx], relation='indication',
                                            save_result=True, return_raw=True, save_name=save)
        print(f'[{name} seed={seed}] eval_disease_centric({idx}) saved to {save}')
    return outs


def scores_of(out):
    # out['prediction'] is keyed by the raw PrimeKG disease id-string, not disease_idx
    # (see try10/try11). One disease per call => exactly one key.
    (preds,) = out['prediction'].values()
    return preds


def rank_in(preds, drug_id):
    ranked = sorted(preds.items(), key=lambda x: -x[1])
    for i, (drug, score) in enumerate(ranked):
        if drug == drug_id:
            return i + 1, len(ranked), score
    return None, len(ranked), None


def report_aggregate(name, cfg, outs_by_seed):
    for idx, dname in cfg['eval'].items():
        print(f"\n=== {name.upper()} @ {dname} ({idx}) over seeds {SEEDS} (attention=True) ===")
        per_seed = [scores_of(o[idx]) for o in outs_by_seed]
        # adopted methodology: average raw scores across seeds, then rank
        ens = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
        zero = {k for k in per_seed[0] if all(p[k] == 0 for p in per_seed)}
        if zero:
            print(f"  exact-0.0 on every seed (isolated in train, unscoreable -- try12): {sorted(zero)}")
        for drug_id, drug_name in REFERENCE_DRUGS[name].items():
            ranks = [rank_in(p, drug_id)[0] for p in per_seed]
            r, total, s = rank_in(ens, drug_id)
            tag = '  [UNSCOREABLE: exact 0.0]' if drug_id in zero else ''
            print(f"  {drug_name:<20} per-seed ranks {ranks} -> ENSEMBLE rank {r}/{total} "
                  f"(mean score {s:.4f}){tag}")


only_disease = os.environ.get('DISEASE')
targets = {only_disease: DISEASES[only_disease]} if only_disease else DISEASES

for disease_name, cfg in targets.items():
    outs = []
    for seed in SEEDS:
        result_dir = os.path.join(RESULTS_DIR, disease_name, f'seed_{seed}')
        outs.append(run_disease_eval(disease_name, cfg, seed, result_dir))
    report_aggregate(disease_name, cfg, outs)
    with open(os.path.join(RESULTS_DIR, disease_name, 'all_seeds_raw.pkl'), 'wb') as f:
        pickle.dump(outs, f)
