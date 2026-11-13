# CPU dry run: build the 120-disease split under the short SplitTag folder name and
# run the script's own check_blind.
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'txgnn_connectivity.py')).read()
ns = {'__file__': os.path.join(HERE, 'txgnn_connectivity.py')}
exec(src.split('for seed in SEEDS:')[0], ns)
d = ns['TxData'](data_folder_path=ns['DATA_DIR'])
d.prepare_split(disease_eval_idx=ns['HOLDOUT'], seed=1)
assert ns['check_blind'](d.df_train, ns['HOLDOUT'], ns['NAME'])
t = d.df_test
print(f"held-out diseases: {len(ns['HOLDOUT'])}; diseases present in test: {t[t.y_type=='disease'].y_idx.nunique()}; "
      f"test rows by relation: {t.relation.value_counts().to_dict()}")
print('split folder:', [x for x in os.listdir(ns['DATA_DIR']) if 'connectivity' in x])
