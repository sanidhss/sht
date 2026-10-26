# CPU dry run: build the split, run the script's own check_blind, then re-run try14's
# neighbour-ring audit against this split.
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'txgnn_gaucher_no_np.py')).read()
ns = {}
exec(src.split("only_disease = os.environ.get('DISEASE')")[0], ns)
cfg = ns['DISEASES']['gaucher_no_np']
d = ns['TxData'](data_folder_path=ns['DATA_DIR'])
d.prepare_split(disease_eval_idx=cfg['holdout'], seed=1)
assert ns['check_blind'](d.df_train, cfg['holdout'], 'gaucher_no_np')

audit = open(os.path.join(HERE, '..', 'try14_26-10-2026_leakage-audit', 'audit_neighbour_leakage.py')).read()
audit = audit.replace("TARGETS = {", "TARGETS = {\n    'gaucher_no_np': ('[16430.0, 1490.0]', [16430.], ['DB00419', 'DB09039', 'DB06720', 'DB04878']),\n} or {", 1)
exec(audit, {'__name__': '__main__'})
