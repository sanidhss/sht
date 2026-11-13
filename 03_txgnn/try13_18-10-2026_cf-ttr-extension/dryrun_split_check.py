# CPU dry run before GPU submission: build the real splits via TxData (cached to
# primekg/disease_eval_[...]), run the script's own check_blind on them, and exercise
# report_aggregate on try11's saved Gaucher eval.pkl files.
import os
import pickle
import sys

src = open(os.path.join(os.path.dirname(__file__), 'txgnn_cf_ttr.py')).read()
src = src.split("only_disease = os.environ.get('DISEASE')")[0]  # defs only, no training loop
ns = {}
exec(src, ns)

for name, cfg in ns['DISEASES'].items():
    d = ns['TxData'](data_folder_path=ns['DATA_DIR'])
    d.prepare_split(disease_eval_idx=cfg['holdout'], seed=1)
    assert ns['check_blind'](d.df_train, cfg['holdout'], name), name
    t = d.df_test
    print(f'{name}: test DD rows on held-out diseases = {len(t)}; relations = {t.relation.value_counts().to_dict()}')

g = '/work/hdd/bisa/sshah21/results/txgnn_v2_indication_selection/gaucher'
outs = [{16430.: pickle.load(open(f'{g}/seed_{s}/eval.pkl', 'rb'))} for s in [1, 2, 3]]
ns['REFERENCE_DRUGS']['gaucher'] = {'DB00419': 'miglustat', 'DB00053': 'imiglucerase'}
ns['report_aggregate']('gaucher', {'eval': {16430.: 'Gaucher disease'}}, outs)
