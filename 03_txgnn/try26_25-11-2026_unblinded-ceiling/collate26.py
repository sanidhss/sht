# Job C (CPU). Rank the unblinded scores on the frozen equalized pool and set them beside
# try25's blinded ranks for the same (node, drug).
#
# Same field, same code: the pool hash is verified first, and rank_of()/ensemble() are
# try24's collate.py / try25's rerank_pool.py functions verbatim, so blinded and unblinded
# differ in one thing only -- whether the answer edge was in the training graph.
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

from config26 import (EDGE_AUDIT, HEADLINE, MODELS, NODE_IDS, ORDER, OUT, POOL_DIR, SEEDS,
                      TARGETS, TRY25)

TRY_DIR = os.path.dirname(os.path.abspath(__file__))


def verify_pool():
    path = os.path.join(POOL_DIR, 'pool_v1.json')
    blob = open(path, 'rb').read()
    got = hashlib.sha256(blob).hexdigest()
    want = open(os.path.join(POOL_DIR, 'pool_v1.sha256')).read().split()[0]
    if got != want:
        sys.exit(f'STOP: pool_v1.json sha256 mismatch\n  expected {want}\n  got      {got}\n'
                 'Every arm must score the identical pool or the comparison is void.')
    ids = {r['drugbank_id'] for r in json.loads(blob)}
    print(f'[0] pool verified: {len(ids)} DrugBank ids, sha256 {got}')
    return ids


def rank_of(scores, drug):
    # identical to try24 collate.py / try25 rerank_pool.py
    ids = list(scores)
    v = np.array([scores[k] for k in ids])
    order = np.argsort(-v, kind='stable')
    pos = {ids[j]: i + 1 for i, j in enumerate(order)}
    return pos.get(drug), len(ids)


def ensemble(per_seed):
    return {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}


def restrict(preds, pool_ids):
    return {k: v for k, v in preds.items() if k in pool_ids}


def main():
    pool_ids = verify_pool()
    audit = pd.read_csv(EDGE_AUDIT)
    raw = dict(zip(pd.read_csv(NODE_IDS).disease_idx.astype(float),
                   pd.read_csv(NODE_IDS).raw_id.astype(str)))

    # ---- load every model's per-seed predictions ----
    preds = {}
    for m in ORDER:
        per_seed = []
        for s in SEEDS:
            p = os.path.join(OUT, m, f'seed_{s}', 'targets.pkl')
            if not os.path.exists(p):
                per_seed = None
                break
            per_seed.append(pickle.load(open(p, 'rb'))['prediction'])
        if per_seed is None:
            print(f'[!] {m}: missing seeds -- model dropped from the unblinded table')
            continue
        preds[m] = per_seed

    # ---- one row per (model, node, drug), both fields, seed-ensembled ----
    rows = []
    for _, a in audit.iterrows():
        m, didx, db = a.model, float(a.disease_idx), a.drug_db_id
        if m not in preds or a.edge_in == 'absent':
            continue
        key = raw[didx]
        if any(key not in p for p in preds[m]):
            print(f'[!] {m}: node {didx} ({key}) absent from its predictions -- skipped')
            continue
        full = ensemble([p[key] for p in preds[m]])
        pool = restrict(full, pool_ids)
        rp, npool = rank_of(pool, db)
        rf, nfull = rank_of(full, db)
        rows.append(dict(model=m, model_holdout=str(MODELS[m][1]), eval_node=a.eval_node,
                         disease_idx=didx, target_run=a.target_run, drug=a.drug,
                         drug_db_id=db, edge_in=a.edge_in, unblinded=bool(a.unblinded),
                         rank_pool=rp, n_pool=npool, rank_7957=rf, n_full=nfull,
                         percentile_pool=round(rp / npool, 4) if rp else None))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(TRY_DIR, 'unblinded_all_models.csv'), index=False)

    # ---- headline: median across eligible (edge in TRAIN) models, rule fixed in config26 ----
    U = R[R.unblinded]
    agg = (U.groupby(['target_run', 'eval_node', 'disease_idx', 'drug', 'drug_db_id'])
             .agg(n_models=('rank_pool', 'size'),
                  unblinded_rank_pool=('rank_pool', HEADLINE),
                  best_model_rank=('rank_pool', 'min'),
                  worst_model_rank=('rank_pool', 'max'),
                  unblinded_rank_7957=('rank_7957', HEADLINE))
             .reset_index())
    for c in ('unblinded_rank_pool', 'unblinded_rank_7957'):
        agg[c] = agg[c].round().astype('Int64')

    # ---- join try25's blinded ranks for the same (node, drug) ----
    B = pd.read_csv(TRY25)
    B = B.rename(columns={'eval_node': 'b_node', 'ref_drug_db_id': 'drug_db_id',
                          'rank_pool': 'blinded_rank_pool', 'rank_7957': 'blinded_rank_7957'})
    bmap = {(r.b_node, r.drug_db_id): r for r in B.itertuples()}
    out = []
    for r in agg.itertuples():
        b = bmap.get((r.eval_node, r.drug_db_id))
        if b is None:   # try25 names some nodes differently; fall back to drug + owning run
            cand = [x for x in B.itertuples()
                    if x.drug_db_id == r.drug_db_id and x.disease == r.target_run]
            b = cand[0] if len(cand) == 1 else None
        d = r._asdict()
        d.pop('Index', None)
        d['blinded_rank_pool'] = None if b is None else b.blinded_rank_pool
        d['blinded_rank_7957'] = None if b is None else b.blinded_rank_7957
        d['blinded_best_cos'] = None if b is None else b.best_cos
        d['blinded_reliable'] = None if b is None else b.reliable
        out.append(d)
    H = pd.DataFrame(out)
    H['pool_gain'] = H.blinded_rank_pool - H.unblinded_rank_pool
    H = H.sort_values('unblinded_rank_pool')
    H.to_csv(os.path.join(TRY_DIR, 'blinded_vs_unblinded.csv'), index=False)

    pd.set_option('display.width', 260, 'display.max_colwidth', 40)
    print('\n==== unblinded (answer edge IN the training graph) vs blinded, pool_v1 ====')
    show = H[['eval_node', 'drug', 'n_models', 'unblinded_rank_pool', 'best_model_rank',
              'worst_model_rank', 'blinded_rank_pool', 'pool_gain', 'blinded_reliable']]
    print(show.to_string(index=False))

    print('\n==== per-model unblinded ranks on pool_v1 (blinded cells marked) ====')
    piv = R.pivot_table(index=['eval_node', 'drug'], columns='model', values='rank_pool')
    print(piv.to_string())
    print('\n[note] a cell whose edge_in is valid/test is NOT unblinded; see '
          'unblinded_all_models.csv for the edge_in of every cell.')

    # `tabulate` is not installed in envs/txgnn2, so to_markdown() raises. The CSVs are
    # already written by this point, but the job would still exit 1 with no summary.
    def table(df, index):
        try:
            return df.to_markdown(index=index)
        except ImportError:
            return '```\n' + df.to_string(index=index) + '\n```'

    md = os.path.join(TRY_DIR, 'SUMMARY.md')
    with open(md, 'w') as f:
        f.write('# try26 unblinded ceiling vs blinded (pool_v1, seed-ensembled)\n\n')
        f.write(table(show, False))
        f.write('\n\n## Per-model unblinded ranks (pool_v1)\n\n')
        f.write(table(piv, True))
        f.write('\n')
    print(f'\n[C] wrote blinded_vs_unblinded.csv, unblinded_all_models.csv, {md}')


if __name__ == '__main__':
    main()
