# Verify the augmented graph before any GPU time is spent. Exits non-zero (so the
# dependent GPU jobs never start) if:
#   1. any node's TxGNN index differs from the original PrimeKG build, or
#   2. a thesis disease's split is not blind (drug edges of a held-out disease in train), or
#   3. one of a thesis disease's 3 prototype lookalikes (try18 mechanism) carries one of
#      its true drugs, other than the known, legitimate Gaucher -> Niemann-Pick case.
# Splits are built in parallel (one process per disease) and cached, so the GPU jobs
# only load them.
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

ORIG = '/work/hdd/bisa/sshah21/primekg'
AUG = '/work/hdd/bisa/sshah21/primekg_aug'
DD = ['indication', 'contraindication', 'off-label use']
# same holdouts as the baseline runs, except Fabry also holds out its parent category
# 'lysosomal lipid storage disorder' (6521), whose migalastat indication was only kept
# out of training by a lucky validation draw in the baseline (try14)
TARGETS = {
    'fabry':   ([543., 6521.], 543., ['DB05018']),
    'gaucher': ([16430.], 16430., ['DB00419', 'DB09039', 'DB06720', 'DB04878']),
    'cf':      ([16250.], 16250., ['DB08820', 'DB09280', 'DB11712', 'DB15444']),
    'ttr':     ([6669., 6361., 6665., 6008., 4803., 5528.], 6669., ['DB11644']),
}
ALLOWED_PROTO_HITS = {('gaucher', 'Niemann-Pick disease')}  # legitimate sibling, try14/15


def index_check():
    from txgnn import TxData
    TxData(data_folder_path=AUG)  # first use builds AUG/kg_directed.csv
    o =pd.read_csv(os.path.join(ORIG, 'kg_directed.csv'), dtype={'x_id': str, 'y_id': str}, low_memory=False)
    n = pd.read_csv(os.path.join(AUG, 'kg_directed.csv'), dtype={'x_id': str, 'y_id': str}, low_memory=False)
    def idmap(df):
        return pd.concat([df[['x_type', 'x_id', 'x_idx']].set_axis(['t', 'i', 'k'], axis=1),
                          df[['y_type', 'y_id', 'y_idx']].set_axis(['t', 'i', 'k'], axis=1)]).drop_duplicates()
    mo, mn = idmap(o), idmap(n)
    m = mo.merge(mn, on=['t', 'i'], how='outer', suffixes=('_o', '_n'), indicator=True)
    bad = m[(m._merge != 'both') | (m.k_o != m.k_n)]
    print(f'[index] nodes orig={len(mo)} aug={len(mn)} mismatched={len(bad)}; '
          f'edges orig={len(o)} aug={len(n)} (+{len(n) - len(o)})', flush=True)
    return len(bad) == 0


def check_target(name):
    from txgnn import TxData
    holdout, q, true = TARGETS[name]
    d = TxData(data_folder_path=AUG)
    d.prepare_split(disease_eval_idx=holdout, seed=1)
    tr = d.df_train
    leak = tr[tr.relation.isin(DD + ['rev_' + r for r in DD]) &
              (((tr.x_type == 'disease') & tr.x_idx.isin(holdout)) | ((tr.y_type == 'disease') & tr.y_idx.isin(holdout)))]
    msgs, ok = [], len(leak) == 0
    msgs.append(f'[{name}] blind: {ok} ({len(leak)} held-out drug-disease rows in train)')
    # prototype lookalikes (all_nodes_profile = disease_disease + proteins), as in try18
    dis = tr[tr.x_type == 'disease']
    e = dis[(dis.relation == 'disease_disease') | (dis.y_type == 'gene/protein')]
    prof = {i: set(zip(g.y_type, g.y_idx)) for i, g in e.groupby('x_idx')}
    ind = tr[tr.relation == 'rev_indication']
    key_drugs = {k: set(g.y_id.astype(str)) for k, g in ind.groupby('x_idx')}
    qp = prof.get(q, set())
    sims = sorted(((len(qp & prof[k]) / np.sqrt(len(qp) * len(prof[k])), k)
                   for k in key_drugs if k in prof and qp), reverse=True)[:3]
    kg = pd.read_csv(os.path.join(ORIG, 'kg.csv'), usecols=['x_type', 'x_id', 'x_name'], dtype=str)
    dname = dict(kg[kg.x_type == 'disease'][['x_id', 'x_name']].values)
    idx2id = dict(zip(dis.x_idx, dis.x_id.astype(str).str.replace(r'\.0$', '', regex=True)))
    for s, k in sims:
        kname = dname.get(idx2id.get(k), str(k))
        hit = sorted(set(true) & key_drugs[k])
        flag = ''
        if hit:
            if (name, kname) in ALLOWED_PROTO_HITS:
                flag = '  (known legitimate sibling - allowed)'
            else:
                flag, ok = '  <-- LEAK: lookalike carries a true drug', False
        msgs.append(f'[{name}]   lookalike {kname!r} cos={s:.3f} true drugs among its indications: {hit or "none"}{flag}')
    return ok, '\n'.join(msgs)


if __name__ == '__main__':
    if not index_check():
        print('FAIL: node indices changed vs original PrimeKG -- results would not be comparable')
        sys.exit(1)
    with Pool(len(TARGETS)) as pool:
        results = pool.map(check_target, list(TARGETS))
    for _, msg in results:
        print(msg, flush=True)
    if not all(ok for ok, _ in results):
        print('FAIL: leakage found -- GPU jobs will not start')
        sys.exit(1)
    print('PASS: indices unchanged, all thesis splits blind, no new lookalike leaks')
