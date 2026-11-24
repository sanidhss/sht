# Build the shared candidate pool every arm of the final cross-architecture comparison
# will score. Built once, then frozen: pool_v1.json + its sha256 are the contract.
#
# Why: TxGNN can only score PrimeKG drug nodes (DrugBank-keyed, 7,957). The chat and
# Tanimoto arms score chembl_approved.json (3,417, ChEMBL-keyed). Different universes,
# so no current per-arm number is comparable. The pool is their intersection.
#
# Nothing is scored here. Building the pool is the whole job.
import csv
import gzip
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

TRY_DIR = os.path.dirname(os.path.abspath(__file__))
TXGNN_RESULTS = '/work/hdd/bisa/sshah21/results/txgnn_v4_new_disease_set'
LIBRARY = '/work/hdd/bisa/sshah21/results/reasoning_screen/chembl_approved.json'
OUT_DIR = '/work/hdd/bisa/sshah21/results/equalized_pool'
# UniChem whole-source mapping, src_id 1 = ChEMBL -> src_id 2 = DrugBank. One file beats
# thousands of per-ID API calls, and it is a stable, citable artefact.
UNICHEM_URL = 'https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src2.txt.gz'
UNICHEM_CACHE = os.path.join(OUT_DIR, 'unichem_src1src2.txt.gz')

# Reference drugs, by DrugBank ID (never by name -- try14's imatinib-mesylate hard
# negative is exactly what name matching produces). A target whose reference is absent
# cannot be evaluated on this pool.
REFERENCES = {
    'DB05018': ('migalastat', 'Fabry'),
    'DB06742': ('ambroxol', 'Gaucher (chaperone arm)'),
    'DB09280': ('lumacaftor', 'CF'),
    'DB11644': ('tafamidis', 'TTR'),
    'DB00619': ('imatinib', 'chronic eosinophilic leukemia'),
    'DB06595': ('midostaurin', 'systemic mastocytosis'),
    'DB06212': ('tolvaptan', 'NSIAD'),
    'DB00348': ('nitisinone', 'tyrosinemia'),
    'DB01219': ('dantrolene', 'malignant hyperthermia'),
    'DB00360': ('sapropterin', 'phenylketonuria'),
}


def txgnn_universe():
    # Any eval_*.pkl holds the full scored drug set: {raw_disease_id: {drugbank_id: score}}.
    # Every disease scores the same drug universe, so one file defines it; we read one per
    # disease and assert they agree, which is cheap and catches a truncated run.
    sets = {}
    for disease in sorted(os.listdir(TXGNN_RESULTS)):
        d = os.path.join(TXGNN_RESULTS, disease, 'seed_1')
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith('eval_') and f.endswith('.pkl'):
                o = pickle.load(open(os.path.join(d, f), 'rb'))
                (preds,) = o['prediction'].values()
                sets[f'{disease}/{f}'] = set(preds)
                break
    if not sets:
        sys.exit(f'FAIL: no eval pickles under {TXGNN_RESULTS}')
    ref_name, ref = next(iter(sets.items()))
    for k, v in sets.items():
        if v != ref:
            sys.exit(f'FAIL: TxGNN drug universe differs between {ref_name} ({len(ref)}) and {k} ({len(v)})')
    print(f'[1] TxGNN drug universe: {len(ref)} DrugBank ids (identical across {len(sets)} eval files)')
    return ref


def unichem_map():
    if not os.path.exists(UNICHEM_CACHE):
        import urllib.request
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f'[2] downloading UniChem mapping -> {UNICHEM_CACHE}')
        urllib.request.urlretrieve(UNICHEM_URL, UNICHEM_CACHE)
    rows = []
    with gzip.open(UNICHEM_CACHE, 'rt') as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2 and parts[0] and parts[1]:
                rows.append((parts[0], parts[1]))
    print(f'[2] UniChem src1(ChEMBL)->src2(DrugBank): {len(rows)} rows (header: {header.strip()!r})')
    return rows


def one_to_one(rows):
    # UniChem is many-to-many (salts, mixtures, hydrates). Keep only pairs whose ChEMBL id
    # has exactly one DrugBank partner AND whose DrugBank id has exactly one ChEMBL partner.
    # Picking arbitrarily inside an ambiguous group is the try14 imatinib-mesylate error.
    by_chembl, by_db = defaultdict(set), defaultdict(set)
    for c, d in rows:
        by_chembl[c].add(d)
        by_db[d].add(c)
    keep, dropped = {}, []
    for c, ds in by_chembl.items():
        if len(ds) == 1:
            (d,) = ds
            if len(by_db[d]) == 1:
                keep[c] = d
                continue
        for d in sorted(ds):
            dropped.append(dict(reason='chembl_multi' if len(ds) > 1 else 'drugbank_multi',
                                chembl_id=c, drugbank_id=d,
                                chembl_group=';'.join(sorted(by_db[d])),
                                drugbank_group=';'.join(sorted(ds))))
    with open(os.path.join(TRY_DIR, 'dropped_ambiguous.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['reason', 'chembl_id', 'drugbank_id', 'chembl_group', 'drugbank_group'])
        w.writeheader()
        w.writerows(sorted(dropped, key=lambda r: (r['chembl_id'], r['drugbank_id'])))
    n_amb_groups = len({r['chembl_id'] for r in dropped} | {r['drugbank_id'] for r in dropped})
    print(f'[3] 1:1 mappings kept: {len(keep)}; ambiguous pairs dropped: {len(dropped)} '
          f'(touching {n_amb_groups} ids) -> dropped_ambiguous.csv')
    return keep, dropped


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tx = txgnn_universe()
    lib = json.load(open(LIBRARY))
    lib_by_chembl = {r['chembl_id']: r for r in lib}
    print(f'[1] library {os.path.basename(LIBRARY)}: {len(lib)} records, {len(lib_by_chembl)} unique ChEMBL ids')

    rows = unichem_map()
    keep, dropped = one_to_one(rows)

    pool, bad_smiles, no_map = [], [], 0
    for chembl_id, rec in lib_by_chembl.items():
        db = keep.get(chembl_id)
        if db is None:
            no_map += 1
            continue
        if db not in tx:
            continue
        smi = rec.get('canonical_smiles') or rec.get('smiles')
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            bad_smiles.append(dict(chembl_id=chembl_id, drugbank_id=db, name=rec.get('name'), smiles=smi))
            continue
        pool.append(dict(drugbank_id=db, chembl_id=chembl_id, name=rec.get('name'),
                         canonical_smiles=Chem.MolToSmiles(mol)))
    pool.sort(key=lambda r: r['drugbank_id'])   # byte-stable file
    print(f'[4] pool: {len(pool)} drugs  (library ids with no 1:1 DrugBank map: {no_map}; '
          f'unparseable SMILES dropped: {len(bad_smiles)})')
    for b in bad_smiles:
        print(f'    unparseable SMILES: {b["drugbank_id"]} {b["chembl_id"]} {b["name"]!r} {b["smiles"]!r}')

    # Step 3: reference pre-check, by DrugBank id.
    in_pool = {r['drugbank_id']: r for r in pool}
    db_to_chembl = {d: c for c, d in keep.items()}
    lib_by_name = defaultdict(list)
    for r in lib:
        lib_by_name[(r.get('name') or '').upper()].append(r['chembl_id'])
    ref_rows, missing = [], []
    for db, (name, target) in sorted(REFERENCES.items(), key=lambda kv: kv[1][0]):
        r = in_pool.get(db)
        if r:
            where = 'in pool'
        elif db not in tx:
            where = 'MISSING: not in the TxGNN drug universe'
        elif db not in db_to_chembl:
            where = 'MISSING: no 1:1 UniChem map (ambiguous or absent)'
        else:
            c = db_to_chembl[db]
            same_name = [x for x in lib_by_name.get(name.upper(), []) if x != c]
            where = (f'MISSING: UniChem maps it to {c}, which is not in the library; '
                     f'the library has {", ".join(same_name)} under this name '
                     f'(different ChEMBL record -- NOT substituted)'
                     if same_name else
                     f'MISSING: UniChem maps it to {c}, which is not in the library')
        ref_rows.append((name, db, r['chembl_id'] if r else '-', target, where))
        if not r:
            missing.append((name, db, target, where))
    print('\n[5] reference pre-check')
    for name, db, ch, target, where in ref_rows:
        print(f'    {name:<14} {db:<9} {ch:<14} {target:<32} {where}')
    if missing:
        print('\n    *** TARGETS THAT CANNOT BE EVALUATED ON THIS POOL ***')
        for name, db, target, where in missing:
            print(f'    - {target}: reference {name} ({db}) {where}')
        print('    No salt or analogue is substituted. Fix the pool or drop these targets.')

    pool_path = os.path.join(OUT_DIR, 'pool_v1.json')
    blob = json.dumps(pool, indent=1, sort_keys=True).encode()
    open(pool_path, 'wb').write(blob)
    sha = hashlib.sha256(blob).hexdigest()
    open(os.path.join(OUT_DIR, 'pool_v1.sha256'), 'w').write(f'{sha}  pool_v1.json\n')

    rep = [f'# Equalized candidate pool v1 ({len(pool)} drugs)\n',
           'Built once and frozen. Every arm of the cross-architecture comparison scores this',
           'exact pool, and verifies the sha256 before scoring.\n',
           '## Provenance', f'- TxGNN drug universe: `{TXGNN_RESULTS}/*/seed_1/eval_*.pkl` keys -- **{len(tx)}** DrugBank ids',
           f'- Library: `{LIBRARY}` -- **{len(lib)}** records ({len(lib_by_chembl)} unique ChEMBL ids)',
           f'- Mapping: UniChem whole-source src_id 1 (ChEMBL) -> src_id 2 (DrugBank), {UNICHEM_URL}',
           f'  cached at `{UNICHEM_CACHE}`, {len(rows)} rows\n',
           '## Filtering', '| step | count |', '|---|---|',
           f'| UniChem pairs | {len(rows)} |',
           f'| 1:1 pairs kept | {len(keep)} |',
           f'| ambiguous pairs dropped (salts/mixtures; see `dropped_ambiguous.csv`) | {len(dropped)} |',
           f'| library ChEMBL ids with no 1:1 DrugBank map | {no_map} |',
           f'| dropped: SMILES failed RDKit parse | {len(bad_smiles)} |',
           f'| **final pool** | **{len(pool)}** |\n',
           'Ambiguity rule: a pair is kept only if its ChEMBL id has exactly one DrugBank',
           'partner *and* that DrugBank id has exactly one ChEMBL partner. Arbitrarily picking',
           'one member of an ambiguous group is the same class of error as try14\'s',
           'imatinib-mesylate hard negative.\n',
           '## Reference pre-check', '| drug | drugbank | chembl | target | status |', '|---|---|---|---|---|']
    rep += [f'| {n} | {db} | {ch} | {t} | {w} |' for n, db, ch, t, w in ref_rows]
    if missing:
        rep.append('\n**Targets that CANNOT be evaluated on this pool:**')
        rep += [f'- {t}: reference {n} ({db}) {w}' for n, db, t, w in missing]
        rep.append('\nNo salt or analogue was substituted.')
    else:
        rep.append('\nAll references present: every target is evaluable on this pool.')
    rep += ['\n## Freeze', f'- `{pool_path}`', f'- sha256: `{sha}`',
            '- Records sorted by `drugbank_id`, JSON written with sorted keys, so the file is byte-stable.']
    open(os.path.join(TRY_DIR, 'POOL_REPORT.md'), 'w').write('\n'.join(rep) + '\n')

    print(f'\npool_v1.json: {pool_path}')
    print(f'sha256: {sha}')


if __name__ == '__main__':
    main()
