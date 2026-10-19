# Pre-training check (CPU, no model): for each target's held-out disease set, compute
# every reference drug's degree in the would-be training graph. Degree 0 => isolated
# node => zero embedding => exact-0.0 score (try12). Also lists any remaining
# drug-disease edges from reference drugs to diseases whose name looks related to the
# target (sibling-node leakage that disease_eval would NOT remove).
import re

import pandas as pd

DATA_DIR = '/work/hdd/bisa/sshah21/primekg'
DD = ['indication', 'contraindication', 'off-label use']
TARGETS = {
    # cystic fibrosis is a single PrimeKG node with all 10 CF indications
    'cf': dict(holdout=[16250.], eval=[16250.], related=r'cystic fibrosis',
               drugs=['Ivacaftor', 'Lumacaftor', 'Tezacaftor', 'Elexacaftor', 'Dornase alfa',
                      'Acetylcysteine', 'Cysteine', 'Mannitol', 'Adenosine phosphate', 'Nafamostat']),
    # tafamidis is indicated for 4 amyloidosis nodes; holding out only ATTRV122I (the
    # only TTR-specific one) would leave tafamidis->'amyloidosis (disease)' in train
    'ttr': dict(holdout=[6669., 6361., 6665., 6008., 4803., 5528.], eval=[6669., 4803., 5528.],
                related=r'amyloid|transthyretin',
                drugs=['Tafamidis', 'Diflunisal', 'Patisiran', 'Inotersen', 'Vutrisiran',
                       'Acoramidis', 'Eplontersen', 'Acetylcysteine', 'Cysteine']),
}

kg = pd.read_csv(f'{DATA_DIR}/kg.csv', low_memory=False,
                 usecols=['x_type', 'x_id', 'x_name', 'relation', 'y_type', 'y_id', 'y_name'])
kd = pd.read_csv(f'{DATA_DIR}/kg_directed.csv', low_memory=False)
norm = lambda s: s.astype(str).str.replace(r'\.0$', '', regex=True)
for df in (kg, kd):
    df['x_id'], df['y_id'] = norm(df.x_id), norm(df.y_id)
drug_name = dict(kg[kg.x_type == 'drug'][['x_name', 'x_id']].drop_duplicates().values)
dis_name = dict(pd.concat([kg[kg.x_type == 'disease'][['x_id', 'x_name']].set_axis(['i', 'n'], axis=1),
                           kg[kg.y_type == 'disease'][['y_id', 'y_name']].set_axis(['i', 'n'], axis=1)]).values)
dis_idx2id = dict(kd[kd.y_type == 'disease'][['y_idx', 'y_id']].drop_duplicates().values)

for t, cfg in TARGETS.items():
    print(f'\n===== {t.upper()}  holdout={cfg["holdout"]} =====')
    print('held-out nodes:', [dis_name.get(dis_idx2id.get(i), '?') for i in cfg['holdout']])
    print('eval nodes:    ', [dis_name.get(dis_idx2id.get(i), '?') for i in cfg['eval']])
    # mirrors disease_eval_fold: drop all DD edges whose disease (y_idx) is held out
    train = kd[~(kd.relation.isin(DD) & kd.y_idx.isin(cfg['holdout']))]
    for name in cfg['drugs']:
        did = drug_name.get(name)
        if did is None:
            print(f'  {name:<20} NOT A PrimeKG DRUG NODE')
            continue
        e = train[((train.x_type == 'drug') & (train.x_id == did)) | ((train.y_type == 'drug') & (train.y_id == did))]
        rels = e.relation.value_counts().to_dict()
        dd_left = e[e.relation.isin(DD) & (e.y_type == 'disease')]
        sib = sorted({(r, dis_name.get(d, d)) for r, d in dd_left[['relation', 'y_id']].values
                      if re.search(cfg['related'], dis_name.get(d, ''), re.I)})
        flag = 'ISOLATED -> unscoreable' if len(e) == 0 else ''
        print(f'  {name:<20} {did:<8} train-degree={len(e):<5} {rels} {flag}')
        if sib:
            print(f'      !! related-disease edges still in train: {sib}')
