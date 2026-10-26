# Option 1: per-drug popularity correction (CPU, no training).
# corrected(drug, D) = (score(drug, D) - mean_bg(drug)) / std_bg(drug), where the
# background is the SAME model's scores for drug across many other diseases. A drug
# then ranks high only if it is unusually high for D specifically -- which removes
# generic hubs (steroids, avelumab, ...) and drugs high for a whole category (try17).
# All scores are seed-ensembled first (mean raw score over seeds), as everywhere else.
# Drugs that are exact 0.0 for every disease are isolated in train (try12): unscoreable,
# ranked last.
#   thesis mode: background = background_150.pkl from background_infer.py
#   sweep mode : leave-one-out background = the other 119 sweep diseases
import os
import pickle

import numpy as np
import pandas as pd

R = '/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection/'
SEEDS = [1, 2, 3]
kg = pd.read_csv('/work/hdd/bisa/sshah21/primekg/kg.csv', low_memory=False, usecols=['x_type', 'x_id', 'x_name'])
name = dict(kg[kg.x_type == 'drug'][['x_id', 'x_name']].astype(str).values)

THESIS = {  # run -> (eval pkl, true drugs to track)
    'fabry': ('eval.pkl', ['DB05018']),
    'gaucher': ('eval.pkl', ['DB00419', 'DB09039', 'DB06720', 'DB04878']),
    'gaucher_no_np': ('eval_16430.pkl', ['DB00419', 'DB09039', 'DB06720', 'DB04878']),
    'cf': ('eval_16250.pkl', ['DB15444', 'DB11712', 'DB09280', 'DB08820', 'DB06151', 'DB00151']),
    'ttr': ('eval_6669.pkl', ['DB11644', 'DB00861', 'DB14582', 'DB14713']),
}
SWEEPS = ['connectivity_sweep', 'connectivity_sweep_profile_more',          # try16, try21
          'connectivity_sweep_learn', 'connectivity_sweep_learn_profile_more']  # try22 arms


def ensemble(pred_dicts):
    drugs = list(pred_dicts[0])
    return drugs, np.array([[p[d] for d in drugs] for p in pred_dicts]).mean(axis=0)


def ranks(scores):
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind='stable')
    r = np.empty(len(scores), int)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def correct(target, bg_matrix):
    # bg_matrix: (n_bg_diseases, n_drugs)
    mu, sd = bg_matrix.mean(axis=0), bg_matrix.std(axis=0)
    z = (target - mu) / (sd + 1e-6)
    isolated = (target == 0) & (np.abs(bg_matrix).max(axis=0) == 0)
    z[isolated] = np.nan
    return z


def load_seeds(run, fname):
    paths = [os.path.join(R, run, f'seed_{s}', fname) for s in SEEDS]
    if not all(os.path.exists(p) for p in paths):
        return None
    return [pickle.load(open(p, 'rb')) for p in paths]


print('==================== THESIS DISEASES (background: 150 diseases, same model) ====================')
for run, (eval_pkl, true) in THESIS.items():
    ev, bg = load_seeds(run, eval_pkl), load_seeds(run, 'background_150.pkl')
    if ev is None or bg is None:
        print(f'\n[{run}] not available yet (eval: {ev is not None}, background: {bg is not None})')
        continue
    (key,) = ev[0]['prediction'].keys()
    drugs, raw = ensemble([o['prediction'][key] for o in ev])
    bg_keys = [k for k in bg[0]['prediction'] if k != key]
    bg_mat = np.array([ensemble([o['prediction'][k] for o in bg])[1] for k in bg_keys])
    z = correct(raw, bg_mat)
    r0, r1 = ranks(raw), ranks(z)
    pos = {d: i for i, d in enumerate(drugs)}
    print(f'\n[{run}]  background diseases: {len(bg_keys)}')
    for d in true:
        print(f'   {name.get(d, d):<22} raw rank {r0[pos[d]]:>5}  ->  corrected rank {r1[pos[d]]:>5}')
    print('   top 10 raw      :', [name.get(drugs[i], drugs[i]) for i in np.argsort(r0)[:10]])
    print('   top 10 corrected:', [name.get(drugs[i], drugs[i]) for i in np.argsort(r1)[:10]])

print('\n==================== 120-DISEASE SWEEPS (leave-one-out background) ====================')
for sweep in SWEEPS:
    outs = load_seeds(sweep, 'eval_all.pkl')
    if outs is None:
        print(f'\n[{sweep}] not available yet')
        continue
    keys = list(outs[0]['prediction'])
    drugs = list(outs[0]['prediction'][keys[0]])
    M = np.array([ensemble([o['prediction'][k] for o in outs])[1] for k in keys])  # (diseases, drugs)
    rows = []
    for i, k in enumerate(keys):
        lab = outs[0]['label'][k]
        isolated = np.all([[o['prediction'][k][d] == 0 for d in drugs] for o in outs], axis=0)
        pos = [j for j, d in enumerate(drugs) if lab.get(d) == 1 and not isolated[j]]
        if not pos:
            continue
        z = correct(M[i], np.delete(M, i, axis=0))
        for tag, sc in [('raw', M[i]), ('corrected', z)]:
            pct = ranks(sc)[pos] / len(drugs)
            rows.append(dict(disease=k, mode=tag, n_pos=len(pos), median_pct=np.median(pct),
                             best_cal=1 - (1 - pct.min()) ** len(pos)))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index='disease', columns='mode', values=['median_pct', 'best_cal'])
    print(f'\n[{sweep}]  diseases: {piv.shape[0]}   (lower = better; chance ~0.50)')
    for m in ['median_pct', 'best_cal']:
        a, b = piv[(m, 'raw')], piv[(m, 'corrected')]
        print(f'   {m:<10} median raw {a.median():.3f} -> corrected {b.median():.3f};  '
              f'improved for {(b < a).mean():.0%} of diseases')
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'rerank_{sweep}.csv'), index=False)
