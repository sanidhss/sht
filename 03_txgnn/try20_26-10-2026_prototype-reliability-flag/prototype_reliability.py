# Option 2: a reliability flag per held-out disease = its best prototype similarity.
# TxGNN paper (Huang et al. 2024): similarity > 0.2 "suggest[s] similar disease
# mechanisms". Computed under both signatures the installed library offers
# (txgnn/model.py ~L83-88):
#   all_nodes_profile      (our config): disease_disease + proteins
#   all_nodes_profile_more (matches the paper's 4-node-type description):
#                          + phenotypes + exposures
# Cosine of one-hot neighbour profiles, over diseases with >=1 training indication
# (the prototype key set), from each run's TRAINING split. CPU only.
import numpy as np
import pandas as pd

P = '/work/hdd/bisa/sshah21/primekg/'
THRESH = 0.2
TARGETS = {
    'Fabry': ('[543.0]', 543.), 'Gaucher': ('[16430.0]', 16430.),
    'Gaucher (NP held out)': ('[16430.0, 1490.0]', 16430.),
    'CF': ('[16250.0]', 16250.), 'TTR (ATTRV122I)': ('[6669.0, 6361.0, 6665.0, 6008.0, 4803.0, 5528.0]', 6669.),
    'TTR (wild type)': ('[6669.0, 6361.0, 6665.0, 6008.0, 4803.0, 5528.0]', 5528.),
    'TTR (hereditary)': ('[6669.0, 6361.0, 6665.0, 6008.0, 4803.0, 5528.0]', 4803.),
}
SIGS = {'all_nodes_profile': {'gene/protein'},
        'all_nodes_profile_more': {'gene/protein', 'effect/phenotype', 'exposure'}}
kg = pd.read_csv(P + 'kg.csv', low_memory=False, usecols=['x_type', 'x_id', 'x_name'])
dname = dict(kg[kg.x_type == 'disease'][['x_id', 'x_name']].astype(str).values)


def best_prototypes(tr, q, other_types, k=3):
    dis = tr[tr.x_type == 'disease']
    e = dis[(dis.relation == 'disease_disease') | dis.y_type.isin(other_types)]
    prof = {i: set(zip(g.y_type, g.y_idx)) for i, g in e.groupby('x_idx')}
    keys = set(tr[tr.relation == 'rev_indication'].x_idx)
    qp = prof.get(q, set())
    sims = sorted(((len(qp & prof[kk]) / np.sqrt(len(qp) * len(prof[kk])), kk)
                   for kk in keys if kk in prof and qp and kk != q), reverse=True)[:k]
    return len(qp), sims


cache = {}
rows = []
for t, (sp, q) in TARGETS.items():
    if sp not in cache:
        tr = pd.read_csv(P + f'disease_eval_{sp}/train.csv', low_memory=False)
        cache[sp] = tr
    tr = cache[sp]
    idx2id = dict(zip(tr[tr.x_type == 'disease'].x_idx, tr[tr.x_type == 'disease'].x_id.astype(str).str.replace(r'\.0$', '', regex=True)))
    for sig, types in SIGS.items():
        n, sims = best_prototypes(tr, q, types)
        best = sims[0][0] if sims else 0.0
        rows.append(dict(target=t, signature=sig, profile_size=n, best_cos=round(best, 3),
                         reliable=best > THRESH,
                         prototypes='; '.join(f'{dname.get(idx2id.get(k), k)} ({s:.2f})' for s, k in sims)))
out = pd.DataFrame(rows)
pd.set_option('display.width', 250, 'display.max_colwidth', 120)
print(out.to_string(index=False))
out.to_csv('prototype_reliability.csv', index=False)
