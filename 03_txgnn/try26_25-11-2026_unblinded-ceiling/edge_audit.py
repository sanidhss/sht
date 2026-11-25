# Job A (CPU). For every (model, target disease node, reference drug), find which file of
# that model's cached split holds the indication edge: train / valid / test / absent.
#
# This is the whole basis of try26, so it is checked explicitly rather than assumed from
# the holdout list. Only 'train' means the model actually saw the edge during training;
# 'valid' and 'test' edges are kept out of the message-passing graph, so a target sitting
# in valid under some model is still blind there and must not be counted as unblinded.
# 'absent' means PrimeKG has no such indication edge at all for that node, under any split.
import os

import pandas as pd

from config26 import DATA_DIR, EDGE_AUDIT, MODELS, NODE_IDS, OUT, TARGETS


def split_folder(holdout):
    return 'disease_eval_' + str([float(i) for i in holdout])


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = [(didx, db) for didx, (_, _, refs) in TARGETS.items() for db in refs]
    rows, dmap = [], {}
    for model, (_, holdout) in MODELS.items():
        folder = os.path.join(DATA_DIR, split_folder(holdout))
        if not os.path.isdir(folder):
            raise SystemExit(f'STOP: split folder missing for model {model}: {folder}')
        where = {}
        for part in ('train', 'valid', 'test'):
            d = pd.read_csv(os.path.join(folder, f'{part}.csv'), low_memory=False)
            dis = d[d.x_type == 'disease']
            for didx, rid in zip(dis.x_idx.astype(float), dis.x_id.astype(str)):
                if didx in TARGETS:
                    dmap[didx] = rid
            d = d[d.relation == 'indication']
            for didx, db in zip(d.y_idx.astype(float), d.x_id.astype(str)):
                where[(didx, db)] = part
        for didx, db in pairs:
            owner, node, refs = TARGETS[didx]
            rows.append(dict(model=model, target_run=owner, eval_node=node, disease_idx=didx,
                             drug_db_id=db, drug=refs[db],
                             edge_in=where.get((didx, db), 'absent'),
                             held_out=didx in [float(i) for i in holdout]))
    # x_idx -> raw PrimeKG id string. eval_disease_centric() keys its prediction dict by
    # that raw id (try10: a third id system, e.g. '15687' or '9266_9265_9267_18150'), so
    # job C needs the map to attribute scores back to a target node. Taken from the split
    # file itself rather than node.csv, because x_idx is TxGNN's per-node-type index.
    miss = [i for i in TARGETS if i not in dmap]
    if miss:
        raise SystemExit(f'STOP: no raw id found for target x_idx {miss}')
    pd.DataFrame([dict(disease_idx=i, raw_id=dmap[i], eval_node=TARGETS[i][1])
                  for i in sorted(TARGETS)]).to_csv(NODE_IDS, index=False)

    R = pd.DataFrame(rows)
    R['unblinded'] = R.edge_in == 'train'
    R.to_csv(EDGE_AUDIT, index=False)

    pd.set_option('display.width', 250)
    print('==== where each target indication edge sits in each model\'s split ====')
    print(R.pivot_table(index=['target_run', 'disease_idx', 'drug'], columns='model',
                        values='edge_in', aggfunc='first').to_string())

    absent = R[R.edge_in == 'absent'].groupby(['eval_node', 'drug']).size()
    if len(absent):
        print('\n[!] no indication edge anywhere in PrimeKG for these (node, drug) pairs -- '
              'they cannot be evaluated blinded OR unblinded:')
        print(absent.to_string())

    n = R[R.unblinded].groupby(['eval_node', 'drug']).size()
    print('\n==== eligible unblinded models per (node, drug) ====')
    print(n.to_string())
    if n.empty:
        raise SystemExit('STOP: no unblinded cell at all -- nothing for job B to score.')
    print(f'\n[A] wrote {EDGE_AUDIT} and {NODE_IDS}')


if __name__ == '__main__':
    main()
