# Job A (CPU). Steps 1-2 of try24: resolve ids, build holdout splits, audit blindness and
# the prototype path. Writes resolution_report.md + audit.out here and resolved.json to
# RESULTS_DIR. Exits 1 on any unresolved id, blindness failure, unhandled answer-carrying
# neighbour/near-duplicate, or prototype key carrying a reference drug.
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import (DATA_DIR, DISEASES, NEAR_DUP_COS, ORDER, RESOLVED, RESULTS_DIR, SIM,
                    TRY_DIR, UNRELIABLE_COS)

ANSWER = ['indication', 'off-label use']
DD = ['indication', 'contraindication', 'off-label use']
# 4-type signature, matching txgnn/model.py for sim_measure='all_nodes_profile_more'
PROFILE_RELS = ['disease_disease', 'rev_disease_protein', 'disease_phenotype_positive', 'rev_exposure_disease']
norm = lambda s: s.astype(str).str.replace(r'\.0$', '', regex=True)


def load_graph():
    kd = pd.read_csv(os.path.join(DATA_DIR, 'kg_directed.csv'), dtype={'x_id': str, 'y_id': str}, low_memory=False)
    kd['x_id'], kd['y_id'] = norm(kd.x_id), norm(kd.y_id)
    nodes = pd.read_csv(os.path.join(DATA_DIR, 'node.csv'), sep='\t', dtype=str)
    return kd, nodes


def both_dirs(kd):
    # undirected view: every edge as (a -> b)
    a = kd[['x_type', 'x_id', 'x_idx', 'relation', 'y_type', 'y_id', 'y_idx']]
    b = a.rename(columns={'x_type': 'y_type', 'x_id': 'y_id', 'x_idx': 'y_idx',
                          'y_type': 'x_type', 'y_id': 'x_id', 'y_idx': 'x_idx'})
    return pd.concat([a, b[a.columns]], ignore_index=True)


def full_profiles(u):
    dis = u[u.x_type == 'disease']
    e = dis[dis.y_type.isin(['disease', 'gene/protein', 'effect/phenotype', 'exposure']) &
            dis.relation.isin(['disease_disease', 'disease_protein', 'disease_phenotype_positive', 'exposure_disease'])]
    return {i: set(zip(g.y_type, g.y_id)) for i, g in e.groupby('x_id')}


def cos(a, b):
    return len(a & b) / np.sqrt(len(a) * len(b)) if a and b else 0.0


def step1(kd, nodes):
    u = both_dirs(kd)
    dis_idx = dict(zip(u[u.x_type == 'disease'].x_id, u[u.x_type == 'disease'].x_idx))
    deg = u[u.x_type == 'disease'].groupby('x_id').size()
    phen = u[(u.x_type == 'disease') & (u.relation == 'disease_phenotype_positive')].groupby('x_id').size()
    dname = dict(nodes[nodes.node_type == 'disease'][['node_id', 'node_name']].values)
    drug_by_name = {}
    for i, n in nodes[nodes.node_type == 'drug'][['node_id', 'node_name']].values:
        drug_by_name.setdefault(n.lower(), []).append(i)
    ans = u[(u.x_type == 'drug') & u.relation.isin(ANSWER) & (u.y_type == 'disease')]
    carriers = ans.groupby('x_id').y_id.apply(set).to_dict()          # drug -> diseases with answer edge
    dd = u[u.relation == 'disease_disease']
    dd_nb = dd.groupby('x_id').y_id.apply(set).to_dict()
    prof = full_profiles(u)

    rep, resolved, problems = ['# try24 resolution report\n'], {}, []
    for key in ORDER:
        cfg = DISEASES[key]
        rep.append(f'\n## {key}: {cfg["label"]}\n')
        m = nodes[(nodes.node_type == 'disease') & nodes.node_name.str.contains(cfg['name_regex'], case=False, regex=True)]
        rep.append('| raw id | internal idx | name | degree | phenotype edges |\n|---|---|---|---|---|')
        for i, n in m[['node_id', 'node_name']].values:
            rep.append(f'| {i} | {dis_idx.get(i, "NOT IN GRAPH")} | {n} | {deg.get(i, 0)} | {phen.get(i, 0)} |')
        ok = True
        for rid, expect in cfg['eval'].items():
            if rid not in set(m.node_id) or dname.get(rid) != expect or rid not in dis_idx:
                problems.append(f'{key}: eval node {rid} ({expect!r}) not resolved (name in graph: {dname.get(rid)!r})')
                ok = False
        for rid in cfg['holdout_extra']:
            if rid not in dis_idx:
                problems.append(f'{key}: holdout node {rid} not in graph')
                ok = False
        # reference drugs
        refs, evaluable_refs = {}, {}
        rep.append('\n| ref drug | DrugBank id | answer edge (indication/off-label) to eval node(s) |\n|---|---|---|')
        for dn in cfg['ref']:
            ids = drug_by_name.get(dn.lower(), [])
            if len(ids) != 1:
                problems.append(f'{key}: drug {dn} resolves to {ids}')
                ok = False
                continue
            db = ids[0]
            refs[db] = dn
            hits = [rid for rid in cfg['eval'] if rid in carriers.get(db, set())]
            rep.append(f'| {dn} | {db} | {", ".join(hits) if hits else "NONE - not evaluable for this disease"} |')
            if hits:
                evaluable_refs[db] = dn
        primary_db = next((d for d, n in refs.items() if n == cfg['ref'][0]), None)
        if primary_db not in evaluable_refs:
            rep.append(f'\n**NOT EVALUABLE BY BLINDING:** primary reference drug has no answer edge to the eval node.')
            ok = False
        # answer carriers near the query: MONDO neighbours or near-duplicates carrying a ref drug
        holdout_raw = set(cfg['eval']) | set(cfg['holdout_extra'])
        rep.append('\nOther diseases carrying a reference-drug answer, by similarity to the eval node (4-type profile, full graph):\n')
        rep.append('| disease | raw id | drug | cos | MONDO neighbour of eval node | held out |\n|---|---|---|---|---|---|')
        rows = []
        for db, dn in refs.items():
            for c in carriers.get(db, set()) - set(cfg['eval']):
                s = max(cos(prof.get(c, set()), prof.get(q, set())) for q in cfg['eval'])
                nb = any(c in dd_nb.get(q, set()) for q in cfg['eval'])
                rows.append((s, c, dn, nb))
        for s, c, dn, nb in sorted(rows, reverse=True):
            held = c in holdout_raw
            rep.append(f'| {dname.get(c, c)} | {c} | {dn} | {s:.3f} | {nb} | {held} |')
            if (nb or s >= NEAR_DUP_COS) and not held:
                problems.append(f'{key}: {dname.get(c, c)!r} ({c}) carries {dn}, cos={s:.3f}, '
                                f'MONDO neighbour={nb}, but is NOT held out')
                ok = False
        resolved[key] = dict(
            label=cfg['label'], ok=ok,
            eval={str(float(dis_idx[r])): dict(raw_id=r, name=dname[r]) for r in cfg['eval'] if r in dis_idx},
            holdout=sorted(float(dis_idx[r]) for r in holdout_raw if r in dis_idx),
            holdout_names={str(float(dis_idx[r])): dname.get(r) for r in holdout_raw if r in dis_idx},
            ref=refs, evaluable_ref=evaluable_refs, primary=primary_db)
    return resolved, rep, problems


def audit(args):
    key, info = args
    from txgnn import TxData
    d = TxData(data_folder_path=DATA_DIR)
    d.prepare_split(disease_eval_idx=info['holdout'], seed=1)   # disease_eval split is cached per holdout list, not per seed
    tr = d.df_train
    holdout = info['holdout']
    leak = tr[tr.relation.isin(DD + ['rev_' + r for r in DD]) &
              (((tr.x_type == 'disease') & tr.x_idx.isin(holdout)) | ((tr.y_type == 'disease') & tr.y_idx.isin(holdout)))]
    blind = len(leak) == 0
    dis = tr[tr.x_type == 'disease']
    e = dis[dis.relation.isin(PROFILE_RELS)]
    prof = {i: set(zip(g.y_type, g.y_idx)) for i, g in e.groupby('x_idx')}
    ind = tr[tr.relation == 'rev_indication']
    key_drugs = {k: set(norm(g.y_id)) for k, g in ind.groupby('x_idx')}
    idx2id = dict(zip(dis.x_idx, norm(dis.x_id)))
    out = {}
    for q, ev in info['eval'].items():
        qp = prof.get(float(q), set())
        sims = sorted(((cos(qp, prof[k]), k) for k in key_drugs if k in prof and qp), reverse=True)[:3]
        keys = []
        for s, k in sims:
            hit = sorted(set(info['ref']) & key_drugs[k])
            keys.append(dict(idx=float(k), raw_id=idx2id.get(k), cos=round(float(s), 4), ref_hits=hit))
        out[q] = dict(profile_size=len(qp), best_cos=keys[0]['cos'] if keys else 0.0, keys=keys)
    return key, blind, len(leak), out


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    kd, nodes = load_graph()
    dname = dict(nodes[nodes.node_type == 'disease'][['node_id', 'node_name']].values)
    resolved, rep, problems = step1(kd, nodes)
    del kd
    todo = [(k, v) for k, v in resolved.items() if v['eval'] and v['primary'] in v['evaluable_ref']]
    print(f'auditing {len(todo)} diseases with signature {SIM}', flush=True)
    with Pool(len(todo)) as pool:
        res = pool.map(audit, todo)
    rep.append('\n# Step 2: blindness and prototype audit (real disease_eval split)\n')
    for key, blind, nleak, out in res:
        r = resolved[key]
        r['blind_check_passed'] = blind
        r['proto'] = out
        rep.append(f'\n## {key}: holdout {r["holdout"]} = {list(r["holdout_names"].values())}')
        rep.append(f'- blind: **{blind}** ({nleak} held-out drug-disease rows in train)')
        if not blind:
            problems.append(f'{key}: NOT BLIND ({nleak} rows)')
            r['ok'] = False
        for q, o in out.items():
            flag = '  **UNRELIABLE (best cos < 0.2)**' if o['best_cos'] < UNRELIABLE_COS else ''
            rep.append(f'- eval {r["eval"][q]["name"]} ({q}): profile size {o["profile_size"]}, best cos {o["best_cos"]}{flag}')
            for k in o['keys']:
                tag = ''
                if k['ref_hits']:
                    tag = f'  <-- LEAK: carries {[r["ref"][h] for h in k["ref_hits"]]}'
                    problems.append(f'{key}: prototype {dname.get(k["raw_id"], k["raw_id"])!r} carries {k["ref_hits"]}')
                    r['ok'] = False
                rep.append(f'  - lookalike {dname.get(k["raw_id"], k["raw_id"])!r} (idx {k["idx"]}) cos={k["cos"]}{tag}')
    rep.append('\n# Problems\n' + ('\n'.join(f'- {p}' for p in problems) if problems else 'none'))
    text = '\n'.join(rep)
    print(text, flush=True)
    open(os.path.join(TRY_DIR, 'resolution_report.md'), 'w').write(text + '\n')
    json.dump(resolved, open(RESOLVED, 'w'), indent=1)
    json.dump(resolved, open(os.path.join(TRY_DIR, 'resolved.json'), 'w'), indent=1)
    if problems:
        print('FAIL: see Problems above -- GPU jobs will not start')
        sys.exit(1)
    print('PASS: all ids resolved, all splits blind, no prototype carries a reference drug')
