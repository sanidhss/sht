# Is the model's ranking disease-specific, or roughly the same list for every disease?
# CPU only, reads saved seed-ensembled scores from try11/try13.
import itertools
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = '/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection/'
RUNS = {'Fabry': 'fabry/seed_{s}/eval.pkl', 'Gaucher': 'gaucher/seed_{s}/eval.pkl',
        'CF': 'cf/seed_{s}/eval_16250.pkl', 'TTR(ATTRV122I)': 'ttr/seed_{s}/eval_6669.pkl'}
TRUE = {'Fabry': {'DB05018': 'migalastat'},
        'Gaucher': {'DB00419': 'miglustat', 'DB09039': 'eliglustat', 'DB06720': 'velaglucerase alfa'},
        'CF': {'DB15444': 'elexacaftor', 'DB11712': 'tezacaftor', 'DB09280': 'lumacaftor', 'DB08820': 'ivacaftor'},
        'TTR(ATTRV122I)': {'DB11644': 'tafamidis'}}
kg = pd.read_csv('/work/hdd/bisa/sshah21/primekg/kg.csv', low_memory=False, usecols=['x_type', 'x_id', 'x_name'])
name = dict(kg[kg.x_type == 'drug'][['x_id', 'x_name']].astype(str).values)

ens, rank = {}, {}
for t, f in RUNS.items():
    ps = []
    for s in [1, 2, 3]:
        (p,) = pickle.load(open(R + f.format(s=s), 'rb'))['prediction'].values()
        ps.append(p)
    ens[t] = {d: float(np.mean([p[d] for p in ps])) for d in ps[0]}
    order = sorted(ens[t], key=lambda d: -ens[t][d])
    rank[t] = {d: i + 1 for i, d in enumerate(order)}
    print(f'{t:<15} top 10: {[name.get(d, d) for d in order[:10]]}')

drugs = list(ens['Fabry'])
print('\nSpearman correlation of full 7,957-drug score lists between diseases:')
for a, b in itertools.combinations(RUNS, 2):
    print(f'  {a:<15} vs {b:<15} {spearmanr([ens[a][d] for d in drugs], [ens[b][d] for d in drugs])[0]:.3f}')
print('top-50 overlap:')
for a, b in itertools.combinations(RUNS, 2):
    ta = set(sorted(drugs, key=lambda d: -ens[a][d])[:50]); tb = set(sorted(drugs, key=lambda d: -ens[b][d])[:50])
    print(f'  {a:<15} vs {b:<15} {len(ta & tb)}/50')

print('\ncross-disease ranks (row = true drug of that disease; columns = rank when scored for each disease):')
print(f'  {"drug":<22}' + ''.join(f'{t:>16}' for t in RUNS))
for t, ds in TRUE.items():
    for d, n in ds.items():
        print(f'  {n + " (" + t + ")":<22}' + ''.join(f'{rank[u][d]:>16}' for u in RUNS))
