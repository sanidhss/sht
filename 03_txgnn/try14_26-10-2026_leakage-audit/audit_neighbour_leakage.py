# Leakage audit (CPU, no model). check_blind() only proves the held-out disease
# nodes themselves have no drug edges in train. This checks the next ring out: drug-
# disease edges from each reference drug to diseases that are graph-close to the eval
# node (direct disease_disease neighbour, or sharing >=1 disease_disease neighbour),
# and reports which split each such edge landed in. Message passing uses TRAIN only
# (TxData.py:108 create_dgl_graph(df_train, ...)); VALID edges only affect
# checkpoint selection.
import pandas as pd

P = '/work/hdd/bisa/sshah21/primekg/'
DD = ['indication', 'contraindication', 'off-label use']
TARGETS = {
    'fabry':   ('[543.0]', [543.], ['DB05018']),
    'gaucher': ('[16430.0]', [16430.], ['DB00419', 'DB09039', 'DB06720', 'DB04878']),
    'cf':      ('[16250.0]', [16250.], ['DB08820', 'DB09280', 'DB11712', 'DB15444']),
    'ttr':     ('[6669.0, 6361.0, 6665.0, 6008.0, 4803.0, 5528.0]', [6669., 5528., 4803.], ['DB11644']),
}
norm = lambda s: s.astype(str).str.replace(r'\.0$', '', regex=True)
kg = pd.read_csv(P + 'kg.csv', low_memory=False, usecols=['x_type', 'x_id', 'x_name', 'y_type', 'y_id', 'y_name'])
kg['x_id'], kg['y_id'] = norm(kg.x_id), norm(kg.y_id)
kd = pd.read_csv(P + 'kg_directed.csv', low_memory=False)
dname = dict(pd.concat([kg[kg.y_type == 'disease'][['y_id', 'y_name']].set_axis(['i', 'n'], axis=1),
                        kg[kg.x_type == 'disease'][['x_id', 'x_name']].set_axis(['i', 'n'], axis=1)]).values)
drname = dict(kg[kg.x_type == 'drug'][['x_id', 'x_name']].values)
idx2id = dict(zip(kd[kd.y_type == 'disease'].y_idx, norm(kd[kd.y_type == 'disease'].y_id)))
nm = lambda i: dname.get(idx2id.get(i), str(i))

for t, (sp, eval_idx, drugs) in TARGETS.items():
    splits = {}
    for s in ['train', 'valid', 'test']:
        df = pd.read_csv(P + f'disease_eval_{sp}/{s}.csv', low_memory=False)
        df['x_id'] = norm(df.x_id)
        splits[s] = df
    tr = splits['train']
    ddg = tr[tr.relation == 'disease_disease']
    nbrs = lambda i: set(ddg[ddg.x_idx == i].y_idx)
    print(f'\n===== {t.upper()}')
    for e in eval_idx:
        n1 = nbrs(e)
        for drug in drugs:
            for s, df in splits.items():
                if s == 'test':
                    continue
                hits = df[df.relation.isin(DD) & (df.x_id == drug) & (df.y_type == 'disease')]
                for rel, y in hits[['relation', 'y_idx']].drop_duplicates().values:
                    shared = n1 & nbrs(y)
                    if y in n1 or shared:
                        how = 'DIRECT neighbour' if y in n1 else f'shares {len(shared)}: {sorted(nm(i) for i in shared)[:4]}'
                        print(f'  @{nm(e)!r}: {drname.get(drug, drug)} -{rel}-> {nm(y)!r} in {s.upper()}  ({how})')
