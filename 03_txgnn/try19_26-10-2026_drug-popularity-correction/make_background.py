# Background diseases for the per-drug popularity correction: 150 diseases with >=1
# indication and >=1 non drug-disease edge (so their representation isn't all-zero),
# excluding the thesis targets and their held-out siblings/parents. Fixed seed.
import pandas as pd

P = '/work/hdd/bisa/sshah21/primekg/'
DD = ['indication', 'contraindication', 'off-label use']
EXCLUDE = {543., 16430., 16250., 6669., 6361., 6665., 6008., 4803., 5528., 1490., 6521., 9528.}
kd = pd.read_csv(P + 'kg_directed.csv', low_memory=False)
non_dd = kd[~kd.relation.isin(DD)]
has_edges = set(non_dd[non_dd.x_type == 'disease'].x_idx) | set(non_dd[non_dd.y_type == 'disease'].y_idx)
cand = sorted(set(kd[(kd.relation == 'indication') & (kd.y_type == 'disease')].y_idx) & has_edges - EXCLUDE)
bg = pd.Series(cand).sample(150, random_state=1).sort_values()
pd.DataFrame({'idx': bg.values}).to_csv('background.csv', index=False)
print(f'{len(cand)} candidates -> 150 background diseases written')
