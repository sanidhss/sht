# Diagnose drugs that score exactly 0.0 in eval_disease_centric output.
# CPU-only, reads saved split CSVs + eval.pkl files; no training.
import os
import pickle

import pandas as pd

DATA_DIR = '/work/hdd/bisa/sshah21/primekg'
RESULTS_DIR = '/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection'
DISEASES = {'fabry': 543.0, 'gaucher': 16430.0}
SEEDS = [1, 2, 3]

kg = pd.read_csv(os.path.join(DATA_DIR, 'kg.csv'), low_memory=False)
kg['x_id'] = kg.x_id.astype(str)
kg['y_id'] = kg.y_id.astype(str)
names = dict(kg[kg.x_type == 'drug'][['x_id', 'x_name']].values)


def all_kg_relations(drug):
    e = kg[(kg.x_id == drug) | (kg.y_id == drug)]
    return e.relation.value_counts().to_dict()


for disease, idx in DISEASES.items():
    # direct path join, NOT glob -- the folder name contains '[...]', which glob
    # treats as a character class and silently matches nothing
    split_dir = os.path.join(DATA_DIR, f'disease_eval_[{idx}]')
    splits = {}
    for s in ['train', 'valid', 'test']:
        df = pd.read_csv(os.path.join(split_dir, s + '.csv'), low_memory=False)
        df['x_id'] = df.x_id.astype(str)
        df['y_id'] = df.y_id.astype(str)
        splits[s] = df
    tr = splits['train']
    train_deg = pd.concat([tr[tr.x_type == 'drug'].x_id, tr[tr.y_type == 'drug'].y_id]).value_counts()

    print(f'\n===== {disease.upper()} (disease_eval_idx={idx}) =====')
    for seed in SEEDS:
        with open(os.path.join(RESULTS_DIR, disease, f'seed_{seed}', 'eval.pkl'), 'rb') as f:
            out = pickle.load(f)
        (preds,) = out['prediction'].values()
        zero_score = {d for d, v in preds.items() if v == 0}
        zero_train = {d for d in preds if train_deg.get(d, 0) == 0}
        ranked = sorted(preds.items(), key=lambda x: -x[1])
        zero_ranks = [i + 1 for i, (d, v) in enumerate(ranked) if v == 0]
        print(f'seed {seed}: exact-0.0 drugs = {sorted(zero_score)}')
        print(f'        0-train-edge drugs = {sorted(zero_train)}')
        print(f'        identical sets: {zero_score == zero_train}; '
              f'tied-at-0 rank band: {min(zero_ranks)}-{max(zero_ranks)} / {len(ranked)}')

    print('per-drug detail (edge rows, both directions):')
    for d in sorted(zero_score):
        per_split = {s: int(((df.x_id == d) | (df.y_id == d)).sum()) for s, df in splits.items()}
        print(f'  {d} {names.get(d, "?"):<22} all-PrimeKG={all_kg_relations(d)}  split={per_split}')
