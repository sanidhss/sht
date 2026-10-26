# try20 — option 2: prototype reliability flag

## Idea
The TxGNN paper says disease similarity > 0.2 "suggest[s] similar disease mechanisms".
A held-out disease is scored as 90% a blend of its 3 most similar training diseases
(try18). So report each target's **best prototype similarity** and flag rankings below
0.2 as unreliable. Computed under both signatures the library offers:
`all_nodes_profile` (our config: disease_disease + proteins) and
`all_nodes_profile_more` (adds phenotypes + exposures, matching the paper's
4-node-type description). CPU only: `prototype_reliability.py` / `.out` / `.csv`.

## Result
| target | best cos (2-type) | best cos (4-type) | reliable? (2 / 4) | notes |
|---|---|---|---|---|
| Fabry | 0.687 | 0.400 | yes / yes | 4-type demotes the leaky chylomicron duplicate from #1 to #3 |
| Gaucher | 0.252 | 0.316 | yes / yes | Niemann-Pick stays #1 |
| Gaucher, NP held out | 0.192 | 0.227 | **no** / yes | 4-type adds odd keys (typhoid fever 0.21) |
| CF | 0.226 | 0.141 | yes / **no** | 4-type swaps bronchiectasis for lipid/cholestasis disorders |
| TTR ATTRV122I | 0.035 | 0.034 | **no / no** | profile of only 3 entries |
| TTR wild type | 0.151 | 0.209 | no / yes (barely) | keys still look unrelated (amoebiasis, FMF) |
| TTR hereditary | 0.000 | 0.000 | **no / no** | prototypes are arbitrary |

## Reading
- **TTR's ranking should be reported as unreliable** under either signature. It's not
  a "weak" result, because the model had no meaningful neighbours to borrow from.
- The richer signature (option 3) is **not obviously better**. It helps Fabry (demotes
  the leak) but makes CF worse and adds implausible prototypes elsewhere. Whether it
  improves ranks is measured by try21 (training run), not assumed.
- The paper's 0.2 threshold may have been calibrated on its own signature definition,
  so treat it as a guide, not a hard rule.
