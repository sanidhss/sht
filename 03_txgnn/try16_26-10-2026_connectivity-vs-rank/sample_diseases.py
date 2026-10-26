# Pick a connectivity-stratified sample of diseases to hold out TOGETHER in one
# disease_eval split (one model per seed scores all of them). CPU only.
import numpy as np
import pandas as pd

P = '/work/hdd/bisa/sshah21/primekg/'
DD = ['indication', 'contraindication', 'off-label use']
N_BINS, PER_BIN, SAMPLE_SEED = 6, 20, 0
# thesis targets + their held-out siblings/parents (try13/14/15) -- not part of this test
EXCLUDE = {543., 16430., 16250., 6669., 6361., 6665., 6008., 4803., 5528., 1490., 6521.}

kd = pd.read_csv(P + 'kg_directed.csv', low_memory=False)
kg = pd.read_csv(P + 'kg.csv', low_memory=False, usecols=['x_type', 'x_id', 'x_name', 'y_type', 'y_id', 'y_name'])
norm = lambda s: s.astype(str).str.replace(r'\.0$', '', regex=True)
kg['x_id'], kg['y_id'], kd['x_id'], kd['y_id'] = norm(kg.x_id), norm(kg.y_id), norm(kd.x_id), norm(kd.y_id)
dname = dict(pd.concat([kg[kg.y_type == 'disease'][['y_id', 'y_name']].set_axis(['i', 'n'], axis=1),
                        kg[kg.x_type == 'disease'][['x_id', 'x_name']].set_axis(['i', 'n'], axis=1)]).values)

ind = kd[(kd.relation == 'indication') & (kd.y_type == 'disease')]
non_dd = kd[~kd.relation.isin(DD)]
deg = pd.concat([non_dd[non_dd.x_type == 'disease'].x_idx, non_dd[non_dd.y_type == 'disease'].y_idx]).value_counts()
drug_deg = pd.concat([kd[kd.x_type == 'drug'].x_idx, kd[kd.y_type == 'drug'].y_idx]).value_counts()

rows = []
for d, g in ind.groupby('y_idx'):
    if d in EXCLUDE:
        continue
    # evaluable drugs: indicated drug that keeps >=1 edge besides its edges to this disease
    to_d = kd[((kd.x_type == 'drug') & kd.x_idx.isin(g.x_idx)) & (kd.y_type == 'disease') & (kd.y_idx == d)]
    n_eval = sum(drug_deg.get(x, 0) - (to_d.x_idx == x).sum() > 0 for x in g.x_idx.unique())
    rows.append(dict(idx=d, id=g.y_id.iloc[0], name=dname.get(g.y_id.iloc[0], '?'),
                     degree=int(deg.get(d, 0)), n_ind=g.x_idx.nunique(), n_eval_drugs=int(n_eval)))
cand = pd.DataFrame(rows)
print(f'diseases with >=1 indication: {len(cand)}; degree 0 (unscoreable): {(cand.degree == 0).sum()}; '
      f'no evaluable drug: {(cand.n_eval_drugs == 0).sum()}')
cand = cand[(cand.degree > 0) & (cand.n_eval_drugs > 0)].copy()
cand['bin'] = pd.qcut(np.log10(cand.degree), N_BINS, labels=False, duplicates='drop')
sample = cand.groupby('bin', group_keys=False).apply(lambda g: g.sample(min(PER_BIN, len(g)), random_state=SAMPLE_SEED))
sample = sample.sort_values('degree').reset_index(drop=True)
print(cand.groupby('bin').degree.agg(['min', 'max', 'count']).to_string())
print(f'sampled {len(sample)} diseases; held-out indication edges: {sample.n_ind.sum()}')
print(sample[['idx', 'name', 'degree', 'n_ind', 'n_eval_drugs', 'bin']].to_string())
sample.to_csv('sample.csv', index=False)
