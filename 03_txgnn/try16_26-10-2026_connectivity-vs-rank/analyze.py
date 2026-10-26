# CPU analysis for try16: does a better-connected disease node mean its true drugs
# rank better? Run after txgnn_connectivity.py has produced eval_all.pkl for all seeds.
#
# Outcome per disease (seed-ensembled: mean raw score over seeds, then rank over all
# drugs, the adopted method):
#   median_pct  -- median percentile rank of the disease's true (test) indications.
#                  Primary: ~0.5 under pure chance regardless of how many true drugs
#                  a disease has, so it isn't confounded by label count.
#   best_cal    -- best true drug's percentile p, calibrated for k true drugs as
#                  1-(1-p)^k (chance of a random ranking doing at least as well);
#                  uniform under chance. Secondary.
# Drugs with exact 0.0 on every seed are isolated in train (try12) and are excluded.
#
# Covariates: disease degree (non drug-disease edges; identical in every split since
# disease_eval only removes drug-disease edges), and whether any true drug has an
# indication in TRAIN to a graph-close disease (direct disease_disease neighbour or
# >=1 shared neighbour) -- try14's "treated sibling" path.
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from txgnn.utils import convert2str

HERE = os.path.dirname(os.path.abspath(__file__))
P = '/work/hdd/bisa/sshah21/primekg/'
R = '/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection/'
SEEDS = [1, 2, 3]
DD = ['indication', 'contraindication', 'off-label use']

kd = pd.read_csv(P + 'kg_directed.csv', low_memory=False)
idx2key = dict(zip(kd[kd.y_type == 'disease'].y_idx, kd[kd.y_type == 'disease'].y_id.apply(convert2str)))
idx2key.update(dict(zip(kd[kd.x_type == 'disease'].x_idx, kd[kd.x_type == 'disease'].x_id.apply(convert2str))))
non_dd = kd[~kd.relation.isin(DD)]
deg_by = {}
for side in ['x', 'y']:
    sub = non_dd[non_dd[f'{side}_type'] == 'disease']
    for (i, rel), n in sub.groupby([f'{side}_idx', 'relation']).size().items():
        deg_by.setdefault(i, {}).setdefault(rel, 0)
        deg_by[i][rel] += n
degree = lambda i: sum(deg_by.get(i, {}).values())


def disease_outcome(per_seed_outs, key):
    preds = [o['prediction'][key] for o in per_seed_outs]
    labels = per_seed_outs[0]['label'][key]
    drugs = list(preds[0])
    ens = np.array([np.mean([p[d] for p in preds]) for d in drugs])
    zero = {d for d in drugs if all(p[d] == 0 for p in preds)}
    order = np.argsort(-ens)
    rank = {drugs[j]: r + 1 for r, j in enumerate(order)}
    pos = [d for d in drugs if labels.get(d) == 1 and d not in zero]
    if not pos:
        return None
    pct = np.array([rank[d] / len(drugs) for d in pos])
    k, best = len(pos), pct.min()
    return dict(n_pos=k, n_pos_isolated=sum(labels.get(d) == 1 for d in zero),
                median_pct=float(np.median(pct)), best_pct=float(best),
                best_cal=float(1 - (1 - best) ** k), best_rank=int(round(best * len(drugs))))


def sibling_paths(train, d, pos_drug_keys, drug_key2idx):
    ddg = train[train.relation == 'disease_disease']
    nb = lambda i: set(ddg[ddg.x_idx == i].y_idx)
    n1 = nb(d)
    ind = train[(train.relation == 'indication') & (train.x_type == 'drug') & (train.y_type == 'disease')]
    hits = 0
    for dk in pos_drug_keys:
        ys = ind[ind.x_idx == drug_key2idx.get(dk)].y_idx.unique()
        if any(y in n1 or (n1 & nb(y)) for y in ys):
            hits += 1
    return hits


def load(path_fmt):
    return [pickle.load(open(path_fmt.format(s=s), 'rb')) for s in SEEDS]


drug_key2idx = dict(zip(kd[kd.x_type == 'drug'].x_id.apply(convert2str), kd[kd.x_type == 'drug'].x_idx))

# ---- sweep diseases
sample = pd.read_csv(os.path.join(HERE, 'sample.csv'))
outs = load(R + 'connectivity_sweep/seed_{s}/eval_all.pkl')
train = pd.read_csv(P + 'disease_eval_[connectivity_sweep_120]/train.csv', low_memory=False)
rows = []
for _, r in sample.iterrows():
    key = idx2key[r.idx]
    o = disease_outcome(outs, key)
    if o is None:
        rows.append(dict(idx=r.idx, name=r['name'], degree=r.degree, n_pos=0))
        continue
    pos = [d for d, l in outs[0]['label'][key].items() if l == 1]
    o.update(idx=r.idx, name=r['name'], degree=degree(r.idx),
             n_protein=deg_by.get(r.idx, {}).get('disease_protein', 0),
             n_sibling_paths=sibling_paths(train, r.idx, pos, drug_key2idx))
    rows.append(o)
res = pd.DataFrame(rows)
no_pos = res[res.n_pos == 0]
res = res[res.n_pos > 0].copy()
res['sibling'] = res.n_sibling_paths > 0
res['log_degree'] = np.log10(res.degree)
res.to_csv(os.path.join(HERE, 'results_per_disease.csv'), index=False)

print(f'diseases analysed: {len(res)} (dropped {len(no_pos)} with no evaluable true drug: '
      f'{list(no_pos.name)})')
print(f'chance reference: median_pct ~0.50, best_cal ~0.50\n')
for y in ['median_pct', 'best_cal']:
    rho, p = stats.spearmanr(res.degree, res[y])
    print(f'Spearman(degree, {y}) = {rho:+.3f}  p={p:.2g}  n={len(res)}   (negative = better-connected ranks better)')
print()
res['deg_bin'] = pd.qcut(res.log_degree, 6, labels=False, duplicates='drop')
print(res.groupby('deg_bin').agg(degree_min=('degree', 'min'), degree_max=('degree', 'max'), n=('idx', 'size'),
                                 median_pct=('median_pct', 'median'), best_cal=('best_cal', 'median'),
                                 frac_sibling=('sibling', 'mean')).round(3).to_string())
print()
print('sibling-path split:')
for flag, g in res.groupby('sibling'):
    rho, p = stats.spearmanr(g.degree, g.median_pct) if len(g) > 3 else (np.nan, np.nan)
    print(f'  sibling={flag!s:<5} n={len(g):<4} median of median_pct={g.median_pct.median():.3f}  '
          f'Spearman(degree, median_pct) within group={rho:+.3f} p={p:.2g}')
rho_ds, p_ds = stats.spearmanr(res.degree, res.sibling)
print(f'  Spearman(degree, has-sibling-path) = {rho_ds:+.3f} p={p_ds:.2g}   (are the two explanations confounded?)')
# rank regression: median_pct ~ degree + sibling (+ n_pos), all as ranks
X = np.column_stack([np.ones(len(res)), stats.rankdata(res.degree), res.sibling.astype(float),
                     stats.rankdata(res.n_pos)])
yv = stats.rankdata(res.median_pct)
beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
resid = yv - X @ beta
se = np.sqrt(np.diag(np.sum(resid ** 2) / (len(yv) - X.shape[1]) * np.linalg.inv(X.T @ X)))
print('\nrank regression  rank(median_pct) ~ rank(degree) + sibling + rank(n_pos):')
for nm_, b, s_ in zip(['intercept', 'rank(degree)', 'sibling', 'rank(n_pos)'], beta, se):
    print(f'  {nm_:<14} beta={b:+8.3f}  t={b / s_:+.2f}')

# ---- thesis targets on the same scale (from their own runs)
print('\nthesis targets on the same scale (their own runs, same outcome definitions):')
targets = [('Fabry', 543., R + 'fabry/seed_{s}/eval.pkl'), ('Gaucher', 16430., R + 'gaucher/seed_{s}/eval.pkl'),
           ('Gaucher, Niemann-Pick held out', 16430., R + 'gaucher_no_np/seed_{s}/eval_16430.pkl'),
           ('CF', 16250., R + 'cf/seed_{s}/eval_16250.pkl'), ('TTR (ATTRV122I)', 6669., R + 'ttr/seed_{s}/eval_6669.pkl')]
for name, idx, fmt in targets:
    try:
        o = disease_outcome(load(fmt), idx2key[idx])
    except FileNotFoundError:
        print(f'  {name:<32} (results not available yet)')
        continue
    print(f'  {name:<32} degree={degree(idx):<5} n_pos={o["n_pos"]:<3} median_pct={o["median_pct"]:.3f} '
          f'best_rank={o["best_rank"]} best_cal={o["best_cal"]:.3f}')
