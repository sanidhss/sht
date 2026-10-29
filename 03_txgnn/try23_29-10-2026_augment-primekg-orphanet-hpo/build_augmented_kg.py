"""Augment PrimeKG's kg.csv with Orphanet disease-gene links and HPO disease-phenotype
links, and remove the known Fabry-duplicate error. Runs anywhere (Python 3 + pandas).

Design rules (so TxGNN's internal indices and every existing script stay valid):
  * Only add edges between nodes that ALREADY exist in PrimeKG. TxGNN assigns its
    per-type index by sorting node ids (utils.preprocess_kg), so an unchanged node set
    means unchanged indices (Fabry stays 543, Gaucher 16430, ...).
  * Never add drug-disease edges (indication/contraindication/off-label): those are the
    answers being tested. Only background knowledge (genes, phenotypes) is added.
  * Every edge is written in both directions, as PrimeKG does.

Usage:
  python build_augmented_kg.py --primekg DIR_WITH_kg.csv_node.csv \
      --external DIR_WITH_orphanet/hpo/mondo --out OUT_DIR [--dry-run]
--dry-run writes the report + added/removed edge lists but not the 1 GB kg.csv.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--primekg', required=True)
ap.add_argument('--external', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---------------- PrimeKG
kg = pd.read_csv(os.path.join(a.primekg, 'kg.csv'), dtype=str, low_memory=False)
COLS = list(kg.columns)
assert COLS == ['relation', 'display_relation', 'x_index', 'x_id', 'x_type', 'x_name', 'x_source',
                'y_index', 'y_id', 'y_type', 'y_name', 'y_source'], COLS
nodes = pd.concat([kg[['x_index', 'x_id', 'x_type', 'x_name', 'x_source']].set_axis(['index', 'id', 'type', 'name', 'source'], axis=1),
                   kg[['y_index', 'y_id', 'y_type', 'y_name', 'y_source']].set_axis(['index', 'id', 'type', 'name', 'source'], axis=1)]
                  ).drop_duplicates(['type', 'id'])
node = {(r.type, r.id): r for r in nodes.itertuples(index=False)}

# MONDO number -> PrimeKG disease node id (merged nodes look like '9266_9265_9267_18150')
mondo2node = {}
for r in nodes[nodes.type == 'disease'].itertuples(index=False):
    for part in r.id.split('_'):
        mondo2node.setdefault(str(int(part)), r.id)
gene_by_symbol = {r.name: r.id for r in nodes[nodes.type == 'gene/protein'].itertuples(index=False)}
pheno_ids = set(nodes[nodes.type == 'effect/phenotype'].id)

existing = set(zip(kg.relation, kg.x_id, kg.y_id))

# ---------------- MONDO cross-references (Orphanet / OMIM -> MONDO)
g = json.load(open(os.path.join(a.external, 'mondo', 'mondo.json')))['graphs'][0]['nodes']
xref2mondo = defaultdict(set)
replaced_by = {}  # obsolete MONDO -> replacement MONDO
for n in g:
    nid = n.get('id', '')
    if 'MONDO_' not in nid:
        continue
    m = str(int(nid.rsplit('_', 1)[1]))
    meta = n.get('meta', {})
    for p in meta.get('basicPropertyValues', []):
        if p['pred'].endswith('IAO_0100001') and 'MONDO_' in p['val']:
            replaced_by[m] = str(int(p['val'].rsplit('_', 1)[1]))
    for x in meta.get('xrefs', []):
        pre, _, val = x['val'].partition(':')
        if pre in ('Orphanet', 'OMIM'):
            xref2mondo[f'{pre}:{val}'].add(m)
# PrimeKG predates the current MONDO release: some of its disease terms are now obsolete
# (e.g. hereditary ATTR amyloidosis MONDO:0017132 -> replaced by 0007100). An obsolete
# PrimeKG term inherits its replacement's cross-references (both nodes get the data).
mondo2nodes = defaultdict(set)
for m, nd in mondo2node.items():
    mondo2nodes[m].add(nd)
    if m in replaced_by:
        mondo2nodes[replaced_by[m]].add(nd)
# only unambiguous mappings (external id -> exactly one MONDO term)
xref2node = {k: mondo2nodes[next(iter(v))] for k, v in xref2mondo.items()
             if len(v) == 1 and mondo2nodes.get(next(iter(v)))}

# ---------------- Orphanet disease-gene (product 6)
SKIP_TYPES = {'Candidate gene tested in', 'Biomarker tested in'}
orpha_pairs = set()
for _, el in ET.iterparse(os.path.join(a.external, 'orphanet', 'en_product6.xml')):
    if el.tag != 'Disorder':
        continue
    code = el.findtext('OrphaCode')
    dnodes = xref2node.get(f'Orphanet:{code}', set())
    for assoc in el.iter('DisorderGeneAssociation'):
        status = assoc.findtext('DisorderGeneAssociationStatus/Name')
        atype = assoc.findtext('DisorderGeneAssociationType/Name')
        sym = assoc.findtext('Gene/Symbol')
        if status == 'Assessed' and atype not in SKIP_TYPES and sym in gene_by_symbol:
            orpha_pairs.update((dnode, gene_by_symbol[sym]) for dnode in dnodes)
    el.clear()

# ---------------- HPO disease-phenotype annotations
hpoa = pd.read_csv(os.path.join(a.external, 'hpo', 'phenotype.hpoa'), sep='\t', comment='#', dtype=str)
hpoa = hpoa[(hpoa.aspect == 'P') & (hpoa.qualifier.fillna('') != 'NOT')]
hpo_pairs = set()
for db, hp in zip(hpoa.database_id, hpoa.hpo_id):
    pre, _, val = db.partition(':')
    dnodes = xref2node.get(f"{'Orphanet' if pre == 'ORPHA' else pre}:{val}", set())
    pid = str(int(hp.split(':')[1]))
    if pid in pheno_ids:
        hpo_pairs.update((dnode, pid) for dnode in dnodes)


def both_dirs(rel, disp, d, other_type, other_id):
    dn, on = node[('disease', d)], node[(other_type, other_id)]
    fwd = [rel, disp, dn.index, dn.id, 'disease', dn.name, dn.source, on.index, on.id, other_type, on.name, on.source]
    rev = [rel, disp, on.index, on.id, other_type, on.name, on.source, dn.index, dn.id, 'disease', dn.name, dn.source]
    return [fwd, rev]


new_rows = []
for d, gid in sorted(orpha_pairs):
    if ('disease_protein', d, gid) not in existing and ('disease_protein', gid, d) not in existing:
        new_rows += both_dirs('disease_protein', 'associated with', d, 'gene/protein', gid)
for d, pid in sorted(hpo_pairs):
    if ('disease_phenotype_positive', d, pid) not in existing and ('disease_phenotype_positive', pid, d) not in existing:
        new_rows += both_dirs('disease_phenotype_positive', 'phenotype present', d, 'effect/phenotype', pid)
added = pd.DataFrame(new_rows, columns=COLS)
assert not added.relation.isin(['indication', 'contraindication', 'off-label use']).any()

# ---------------- Targeted fix: chylomicron retention disease (CRD) carries Fabry's data
# In PrimeKG, CRD (MONDO 9528) has all of Fabry's (MONDO 10526) gene links incl. GLA and a
# migalastat indication (try18). Remove the migalastat-CRD edges and CRD's gene links that
# duplicate Fabry's, unless Orphanet independently supports them for CRD.
CRD, FABRY, MIGALASTAT = '9528', '10526', 'DB05018'
dp = kg[kg.relation == 'disease_protein']
genes_of = lambda d: set(dp[dp.x_id == d].y_id) | set(dp[dp.y_id == d].x_id)
dup_genes = (genes_of(CRD) & genes_of(FABRY)) - {gid for d, gid in orpha_pairs if d == CRD}
rm = (((kg.x_id == CRD) & (kg.y_id == MIGALASTAT)) | ((kg.x_id == MIGALASTAT) & (kg.y_id == CRD)) |
      ((kg.relation == 'disease_protein') & (((kg.x_id == CRD) & kg.y_id.isin(dup_genes)) |
                                             ((kg.y_id == CRD) & kg.x_id.isin(dup_genes)))))
removed = kg[rm]

# ---------------- Report
added.to_csv(os.path.join(a.out, 'added_edges.csv'), index=False)
removed.to_csv(os.path.join(a.out, 'removed_edges.csv'), index=False)
WATCH = {'Fabry disease': FABRY, 'Gaucher disease': '9266_9265_9267_18150', 'cystic fibrosis': '9061',
         'ATTRV122I amyloidosis': '19441', 'hereditary ATTR amyloidosis': '17132',
         'wild type ATTR amyloidosis': '18018', 'familial amyloid neuropathy': '7100',
         'chylomicron retention disease': CRD}
lines = [f'MONDO xrefs usable (unambiguous, in PrimeKG): {len(xref2node)}',
         f'Orphanet disease-gene pairs mapped: {len(orpha_pairs)}; HPO disease-phenotype pairs mapped: {len(hpo_pairs)}',
         f'edges added (both directions): {len(added)}  '
         f'[disease_protein {int((added.relation == "disease_protein").sum())}, '
         f'disease_phenotype_positive {int((added.relation == "disease_phenotype_positive").sum())}]',
         f'edges removed (CRD fix): {len(removed)}  [migalastat-CRD + {len(dup_genes)} duplicated gene links]',
         '', 'per-disease edge counts (one direction), before -> after:']
after = pd.concat([kg[~rm], added])
for nm, d in WATCH.items():
    def cnt(df, rel):
        return int(((df.relation == rel) & (df.x_id == d)).sum())
    lines.append(f'  {nm:<30} genes {cnt(kg, "disease_protein"):>3} -> {cnt(after, "disease_protein"):>3}   '
                 f'phenotypes {cnt(kg, "disease_phenotype_positive"):>3} -> {cnt(after, "disease_phenotype_positive"):>3}')
lines.append(f'  hereditary ATTR -> TTR gene linked after: '
             f'{((after.relation == "disease_protein") & (after.x_id == "17132") & (after.y_name == "TTR")).any()}')
open(os.path.join(a.out, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

if not a.dry_run:
    after.to_csv(os.path.join(a.out, 'kg.csv'), index=False)
    print(f'wrote {os.path.join(a.out, "kg.csv")} ({len(after)} rows)')
