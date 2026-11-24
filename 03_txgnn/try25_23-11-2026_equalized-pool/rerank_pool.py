# Re-rank TxGNN's EXISTING predictions on the frozen equalized pool. No training, no GPU.
#
# Only the candidate field changes: the score dicts are the ones already saved by try23
# (augmented + symptom lookalikes) and try24 (6 new diseases). Every key outside the pool
# is dropped, then the same ranking and seed-ensembling as try24's collate.py is applied
# (rank of the mean raw score over seeds 1-3).
#
# Both fields are reported side by side, and the specificity control is re-run on the new
# field: a rank only counts if the drug is not ranked that high under every other disease's
# model. That control is what caught nitisinone on the full field (3,224 for tyrosinemia
# while dantrolene sat at 3,164 on the same model), and a rank in a new field has to earn
# it again. try20's rule carries forward: best lookalike cos < 0.2 reports as UNRELIABLE.
import hashlib
import json
import os
import pickle
import sys
from multiprocessing import Pool as ProcPool

import numpy as np
import pandas as pd

TRY_DIR = os.path.dirname(os.path.abspath(__file__))
POOL_DIR = '/work/hdd/bisa/sshah21/results/equalized_pool'
V3 = '/work/hdd/bisa/sshah21/results/txgnn_v3_augmented_symptoms'     # try23
V4 = '/work/hdd/bisa/sshah21/results/txgnn_v4_new_disease_set'        # try24
AUG = '/work/hdd/bisa/sshah21/primekg_aug'
T24 = os.path.join(os.path.dirname(TRY_DIR), 'try24_10-11-2026_new-disease-set', 'resolved.json')
SEEDS = [1, 2, 3]
UNRELIABLE_COS = 0.2
# 4-type signature, matching txgnn/model.py for sim_measure='all_nodes_profile_more'
PROFILE_RELS = ['disease_disease', 'rev_disease_protein', 'disease_phenotype_positive', 'rev_exposure_disease']

# try23 runs: run -> (split folder name, {eval_idx: (node name, {drug: label})})
V3_RUNS = {
    'fabry':   ('disease_eval_[543.0, 6521.0]',
                {543: ('Fabry disease', {'DB05018': 'migalastat'})}),
    'gaucher': ('disease_eval_[16430.0]',
                {16430: ('Gaucher disease', {'DB00419': 'miglustat', 'DB09039': 'eliglustat'})}),
    'cf':      ('disease_eval_[16250.0]',
                {16250: ('cystic fibrosis', {'DB08820': 'ivacaftor', 'DB09280': 'lumacaftor',
                                             'DB11712': 'tezacaftor', 'DB15444': 'elexacaftor'})}),
    'ttr':     ('disease_eval_[6669.0, 6361.0, 6665.0, 6008.0, 4803.0, 5528.0]',
                {6669: ('ATTRV122I amyloidosis', {'DB11644': 'tafamidis'}),
                 5528: ('wild type ATTR amyloidosis', {'DB11644': 'tafamidis'}),
                 4803: ('hereditary ATTR amyloidosis', {'DB11644': 'tafamidis'})}),
}


def verify_pool():
    path = os.path.join(POOL_DIR, 'pool_v1.json')
    blob = open(path, 'rb').read()
    got = hashlib.sha256(blob).hexdigest()
    want = open(os.path.join(POOL_DIR, 'pool_v1.sha256')).read().split()[0]
    if got != want:
        sys.exit(f'STOP: pool_v1.json sha256 mismatch\n  expected {want}\n  got      {got}\n'
                 'Every arm must score the identical pool or the comparison is void.')
    pool = json.loads(blob)
    ids = {r['drugbank_id'] for r in pool}
    print(f'[0] pool verified: {len(pool)} records, {len(ids)} unique DrugBank ids, sha256 {got}')
    return ids


def rank_of(scores, drug):
    # identical to try24 collate.py rank_of(): descending score, stable order
    ids = list(scores)
    v = np.array([scores[k] for k in ids])
    order = np.argsort(-v, kind='stable')
    pos = {ids[j]: i + 1 for i, j in enumerate(order)}
    return pos.get(drug), len(ids)


def ensemble(per_seed):
    return {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}


def load_seeds(root, disease, idx):
    out = []
    for s in SEEDS:
        p = os.path.join(root, disease, f'seed_{s}', f'eval_{idx}.pkl')
        if not os.path.exists(p):
            return None
        o = pickle.load(open(p, 'rb'))
        (preds,) = o['prediction'].values()   # keyed by raw PrimeKG disease id; one per call
        out.append(preds)
    return out


def restrict(preds, pool_ids):
    return {k: v for k, v in preds.items() if k in pool_ids}


def best_cos_v3(args):
    # Recompute try23's best lookalike similarity from the cached training split, with the
    # same 4-type profile used by try24's audit, so both tables use one definition.
    run, folder, idxs = args
    tr = pd.read_csv(os.path.join(AUG, folder, 'train.csv'), low_memory=False)
    dis = tr[tr.x_type == 'disease']
    e = dis[dis.relation.isin(PROFILE_RELS)]
    prof = {i: set(zip(g.y_type, g.y_idx)) for i, g in e.groupby('x_idx')}
    keys = set(tr[tr.relation == 'rev_indication'].x_idx)
    out = {}
    for idx in idxs:
        qp = prof.get(float(idx), set())
        sims = sorted(len(qp & prof[k]) / np.sqrt(len(qp) * len(prof[k]))
                      for k in keys if k in prof and qp and prof[k])
        out[idx] = round(float(sims[-1]), 4) if sims else 0.0
    return run, out


def main():
    pool_ids = verify_pool()

    rows, ens_rows, models = [], [], {}
    # ---- try24 (6 new diseases) ----
    R = json.load(open(T24))
    for disease, info in R.items():
        if not info.get('ok'):
            continue
        for q, ev in info['eval'].items():
            idx = int(float(q))
            per_seed = load_seeds(V4, disease, idx)
            if per_seed is None:
                print(f'    skip {disease}/{idx}: missing seeds')
                continue
            best = info['proto'][q]['best_cos']
            models[(disease, ev['name'])] = ensemble([restrict(p, pool_ids) for p in per_seed])
            for db, dn in info['evaluable_ref'].items():
                for s, p in zip(SEEDS, per_seed):
                    rp, npool = rank_of(restrict(p, pool_ids), db)
                    rf, nfull = rank_of(p, db)
                    rows.append(dict(source='try24', disease=disease, eval_node=ev['name'],
                                     ref_drug=dn, ref_drug_db_id=db, seed=s,
                                     rank_pool=rp, n_pool=npool,
                                     percentile_pool=round(rp / npool, 4) if rp else None,
                                     rank_7957=rf, n_full=nfull,
                                     percentile_7957=round(rf / nfull, 4) if rf else None,
                                     best_cos=best, reliable=best >= UNRELIABLE_COS, pool='pool_v1'))
                ep, np_ = rank_of(models[(disease, ev['name'])], db)
                ef, nf = rank_of(ensemble(per_seed), db)
                ens_rows.append(dict(source='try24', disease=disease, eval_node=ev['name'], ref_drug=dn,
                                     ref_drug_db_id=db, rank_pool=ep, n_pool=np_, rank_7957=ef, n_full=nf,
                                     best_cos=best, reliable=best >= UNRELIABLE_COS, pool='pool_v1'))

    # ---- try23 (augmented + symptom lookalikes) ----
    with ProcPool(len(V3_RUNS)) as pp:
        cos_by_run = dict(pp.map(best_cos_v3, [(r, f, list(e)) for r, (f, e) in V3_RUNS.items()]))
    for run, (_, evals) in V3_RUNS.items():
        for idx, (node_name, refs) in evals.items():
            per_seed = load_seeds(V3, run, idx)
            if per_seed is None:
                print(f'    skip {run}/{idx}: missing seeds')
                continue
            best = cos_by_run[run][idx]
            models[(run, node_name)] = ensemble([restrict(p, pool_ids) for p in per_seed])
            for db, dn in refs.items():
                for s, p in zip(SEEDS, per_seed):
                    rp, npool = rank_of(restrict(p, pool_ids), db)
                    rf, nfull = rank_of(p, db)
                    rows.append(dict(source='try23', disease=run, eval_node=node_name,
                                     ref_drug=dn, ref_drug_db_id=db, seed=s,
                                     rank_pool=rp, n_pool=npool,
                                     percentile_pool=round(rp / npool, 4) if rp else None,
                                     rank_7957=rf, n_full=nfull,
                                     percentile_7957=round(rf / nfull, 4) if rf else None,
                                     best_cos=best, reliable=best >= UNRELIABLE_COS, pool='pool_v1'))
                ep, np_ = rank_of(models[(run, node_name)], db)
                ef, nf = rank_of(ensemble(per_seed), db)
                ens_rows.append(dict(source='try23', disease=run, eval_node=node_name, ref_drug=dn,
                                     ref_drug_db_id=db, rank_pool=ep, n_pool=np_, rank_7957=ef, n_full=nf,
                                     best_cos=best, reliable=best >= UNRELIABLE_COS, pool='pool_v1'))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TRY_DIR, 'results_pool_v1.csv'), index=False)
    # ---- specificity control on the NEW field ----
    # Every reference drug that is in the pool, ranked under every model, pool-restricted.
    drugs = {(r['ref_drug_db_id'], r['ref_drug']) for r in ens_rows}
    spec = []
    for db, dn in sorted(drugs, key=lambda x: x[1]):
        for (disease, node), sc in models.items():
            r, n = rank_of(sc, db)
            spec.append(dict(drug=dn, drug_db_id=db, model_disease=disease, model_node=node,
                             rank_pool=r, n_pool=n))
    S = pd.DataFrame(spec)
    S.to_csv(os.path.join(TRY_DIR, 'specificity_pool_v1.csv'), index=False)

    # Verdict per (model, drug): a rank counts only if the drug ranks better under its own
    # disease's model than under EVERY other model. Eyeballing the matrix is how a
    # borrowed-category rank slips through, so the comparison is made explicit here.
    by_drug = {}
    for r in spec:
        by_drug.setdefault(r['drug_db_id'], []).append(r)
    for e in ens_rows:
        own = (e['disease'], e['eval_node'])
        others = [r for r in by_drug[e['ref_drug_db_id']]
                  if (r['model_disease'], r['model_node']) != own and r['rank_pool'] is not None]
        best = min(others, key=lambda r: r['rank_pool']) if others else None
        e['best_other_rank'] = best['rank_pool'] if best else None
        e['best_other_model'] = f"{best['model_disease']}/{best['model_node']}" if best else None
        e['specificity_pass'] = (None if e['rank_pool'] is None or best is None
                                 else bool(e['rank_pool'] < best['rank_pool']))
    E = pd.DataFrame(ens_rows)
    E.to_csv(os.path.join(TRY_DIR, 'results_pool_v1_ensemble.csv'), index=False)

    # ---- printed summary ----
    pd.set_option('display.width', 250, 'display.max_colwidth', 44)
    print('\n==== seed-ensembled ranks: pool_v1 vs full field ====')
    show = E.copy()
    show['rank_pool'] = [('not in pool_v1' if r is None else int(r)) for r in show.rank_pool]
    show['verdict'] = np.where(~show.reliable, 'UNRELIABLE (cos<0.2)',
                      np.where(show.specificity_pass.fillna(False), 'counts',
                               'FAILS specificity'))
    print(show[['source', 'disease', 'eval_node', 'ref_drug', 'rank_pool', 'rank_7957',
                'best_cos', 'best_other_rank', 'best_other_model', 'verdict']].to_string(index=False))

    print('\n==== specificity control on pool_v1 (rank of each reference drug under every model) ====')
    piv = S.pivot_table(index=['model_disease', 'model_node'], columns='drug', values='rank_pool')
    print(piv.to_string())

    print('\n[note] own-disease rank is the diagonal; a rank only counts if it is far better '
          'there than under the other models.')


if __name__ == '__main__':
    main()
