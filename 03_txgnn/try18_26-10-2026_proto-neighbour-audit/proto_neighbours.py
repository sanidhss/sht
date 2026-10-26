# Reproduce TxGNN's prototype step for each held-out disease, on CPU, from the actual
# training split. model.py (proto=True, sim_measure='all_nodes_profile',
# agg_measure='rarity', proto_num=3):
#   profile(d) = one-hot of d's disease_disease neighbours ++ one-hot of its proteins,
#                taken from the TRAINING graph G (utils.obtain_disease_profile)
#   keys       = diseases with >=1 edge of the scored relation (rev_indication) in G
#   query rep  = (1-c)*own + c*sum_k w_k*key_k, over the top-3 cosine-similar keys,
#                with w = L1-normalised similarities and
#                c = 0.7*exp(-0.7*deg) + 0.2 (deg = query's rev_indication degree in G)
# For a held-out disease deg = 0, so c = 0.9: its scoring representation is 90% the
# blend of its 3 nearest keys. Lists each target's 3 keys and their training
# indications, and checks whether the true drug is among them (a proto-path leak).
import numpy as np
import pandas as pd

P = '/work/hdd/bisa/sshah21/primekg/'
TARGETS = {
    'Fabry':   ('[543.0]', 543., ['DB05018']),
    'Gaucher': ('[16430.0]', 16430., ['DB00419', 'DB09039', 'DB06720', 'DB04878']),
    'Gaucher (Niemann-Pick held out, try15)': ('[16430.0, 1490.0]', 16430., ['DB00419', 'DB09039', 'DB06720', 'DB04878']),
    'CF':      ('[16250.0]', 16250., ['DB08820', 'DB09280', 'DB11712', 'DB15444']),
    'TTR (ATTRV122I)': ('[6669.0, 6361.0, 6665.0, 6008.0, 4803.0, 5528.0]', 6669., ['DB11644']),
}
norm = lambda s: s.astype(str).str.replace(r'\.0$', '', regex=True)
kg = pd.read_csv(P + 'kg.csv', low_memory=False, usecols=['x_type', 'x_id', 'x_name', 'y_type', 'y_id', 'y_name'])
kg['x_id'], kg['y_id'] = norm(kg.x_id), norm(kg.y_id)
dname = dict(pd.concat([kg[kg.y_type == 'disease'][['y_id', 'y_name']].set_axis(['i', 'n'], axis=1),
                        kg[kg.x_type == 'disease'][['x_id', 'x_name']].set_axis(['i', 'n'], axis=1)]).values)
drname = dict(kg[kg.x_type == 'drug'][['x_id', 'x_name']].values)

for t, (sp, q, true_drugs) in TARGETS.items():
    tr = pd.read_csv(P + f'disease_eval_{sp}/train.csv', low_memory=False)
    tr['x_id'], tr['y_id'] = norm(tr.x_id), norm(tr.y_id)
    idx2id = dict(zip(tr[tr.x_type == 'disease'].x_idx, tr[tr.x_type == 'disease'].x_id))
    nm = lambda i: dname.get(idx2id.get(i), str(i))
    dd = tr[(tr.relation == 'disease_disease')]
    dp = tr[(tr.x_type == 'disease') & (tr.y_type == 'gene/protein')]
    prof = {}
    for i, g in dd.groupby('x_idx'):
        prof.setdefault(i, set()).update(('d', y) for y in g.y_idx)
    for i, g in dp.groupby('x_idx'):
        prof.setdefault(i, set()).update(('p', y) for y in g.y_idx)
    ind = tr[(tr.relation == 'rev_indication')]            # disease -> drug
    keys = set(ind.x_idx)
    qp = prof.get(q, set())
    sims = sorted(((len(qp & prof.get(k, set())) / (np.sqrt(len(qp)) * np.sqrt(len(prof.get(k, set())))), k)
                   for k in keys if prof.get(k) and qp), reverse=True)[:3]
    w = np.array([s for s, _ in sims]); w = w / w.sum() if w.sum() else w
    c = 0.7 * np.exp(-0.7 * (q in keys and len(ind[ind.x_idx == q]))) + 0.2
    print(f'\n===== {t}: query {nm(q)!r}  profile size {len(qp)}  in keys? {q in keys}  blend c={c:.2f}')
    for (s, k), wk in zip(sims, w):
        drugs = ind[ind.x_idx == k].y_id.unique()
        hit = sorted(drname.get(d, d) for d in set(drugs) & set(true_drugs))
        top = sorted(drname.get(d, d) for d in drugs)
        print(f'  key {nm(k)!r:<55} cos={s:.3f} weight={wk:.2f} c*w={c * wk:.2f}  '
              f'#indications={len(drugs)}  TRUE DRUG AMONG THEM: {hit or "no"}')
        print(f'      indications: {top[:14]}{" ..." if len(top) > 14 else ""}')
