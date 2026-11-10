# try14 — ChEMBL target vetting record

Every id below was checked before being written into `TARGETS`, by querying
`target.json?target_components__accession=<UniProt>` and printing `target_type`,
`organism`, `pref_name` and IC50/EC50 record counts, then checking whether the reference
drug actually has measured activity against the accepted id. This is the step the Gaucher
entry exists to enforce: `CHEMBL612545` looked useful but was a bulk-import bucket with
266,949 unrelated records.

## Accepted ids

| target | gene | UniProt | accepted id | type | organism | IC50 | EC50 |
|---|---|---|---|---|---|---|---|
| cel | PDGFRA | P16234 | CHEMBL2007 | SINGLE PROTEIN | Homo sapiens | 1004 | 6 |
| mastocytosis | KIT | P10721 | CHEMBL1936 | SINGLE PROTEIN | Homo sapiens | 4352 | 206 |
| nsiad | AVPR2 | P30518 | CHEMBL1790 | SINGLE PROTEIN | Homo sapiens | 594 | 406 |
| tyrosinemia | HPD | P32754 | CHEMBL1861 | SINGLE PROTEIN | Homo sapiens | 68 | 0 |
| mh | RYR1 | P21817 | CHEMBL1846 | SINGLE PROTEIN | Homo sapiens | 3 | 8 |
| pku | PAH | P00439 | CHEMBL3076 | SINGLE PROTEIN | Homo sapiens | 1 | 0 |

## Rejected ids

All rejected for `target_type`, per the rule. In every case the rejected id held either
zero records or a subset already covered by the accepted id, so nothing was lost.

| rejected id | type | why | records |
|---|---|---|---|
| CHEMBL2095189 | PROTEIN COMPLEX | PDGFR complex, not the alpha subunit alone | 430 IC50 |
| CHEMBL4630733 | PROTEIN-PROTEIN INTERACTION | cereblon/PDGFRA degrader assay | 0 |
| CHEMBL2111428 | SELECTIVITY GROUP | KDR **and** KIT jointly | 0 |
| CHEMBL4523731 | PROTEIN-PROTEIN INTERACTION | VHL/KIT | 0 |
| CHEMBL4630731 | PROTEIN-PROTEIN INTERACTION | cereblon/KIT | 0 |
| CHEMBL2363078 | PROTEIN FAMILY | "Vasopressin receptor" family | 0 |
| CHEMBL4523980 | SELECTIVITY GROUP | V1a/V2 jointly | 0 |
| CHEMBL5482972 | CHIMERIC PROTEIN | GPR27-V2 fusion construct | 27 EC50 |
| CHEMBL4106135 | PROTEIN COMPLEX | RyR1/FKBP12 | 13 EC50 |

No non-Homo-sapiens target was returned for any of the six accessions, because the query
is keyed on the human UniProt accession.

---

# Two places the spec had to change, and why

## 1. The `assay_organism = Homo sapiens` filter would have deleted the entire dataset

The instruction was to filter `assay_organism = Homo sapiens` on tyrosinemia, to avoid the
plant-HPPD herbicide literature — the trap that inflated GAA to 35,883 records when only
151 were human binding assays.

**Measured:** `assay_organism` is `null` on **every** record of **all six** targets.

```
cel          1010 records, assay_organism: {'(none)': 1010}
mastocytosis 4554 records, assay_organism: {'(none)': 4554}
nsiad         982 records, assay_organism: {'(none)': 982}
tyrosinemia    68 records, assay_organism: {'(none)': 68}
mh             11 records, assay_organism: {'(none)': 11}
pku             1 record,  assay_organism: {'(none)': 1}
```

The field is simply not populated on ChEMBL's *activity* endpoint; it lives on the *assay*
endpoint. So `if assay_organism != "Homo sapiens": drop` would have dropped 100% of every
target. The filter as written protects nothing and destroys everything.

**The trap is already avoided, by target selection rather than record filtering.** The
plant HPPD assays live under *different* ChEMBL target ids whose organism is a plant
species. Querying by the human accession P32754 returns exactly one target, CHEMBL1861
(SINGLE PROTEIN / Homo sapiens), so plant data is never fetched. Spot-checking the assay
descriptions confirms it:

```
assay=CHEMBL3412604  "Inhibition of human recombinant His-tagged HPPD expressed in
                      Escherichia coli BL21(DE3)..."
```

Human enzyme, bacterial expression host — which is normal and correct, and would have been
*wrongly* discarded by a naive organism filter looking at the host.

**What was built instead:** `require_human_assay`, which fetches the real `assay_organism`
from the assay endpoint (cached per assay id, so it costs one request per distinct assay,
not per record) and drops only records whose organism is **explicitly** non-human. Nulls
are kept. It is available per target and currently enabled on none of them, because the
target-level organism check already does the job and no target needs it. The guard exists
so the next session has a correct tool if a mixed-organism target ever turns up.

## 2. Three of the six reference drugs cannot be found the way try06 looks for them

try06 matches the reference drug by exact lowercased `molecule_pref_name`. Checked against
the real data:

| drug | found by pref_name? | activities vs its target, **any** type |
|---|---|---|
| imatinib | yes, 13 records | fine |
| midostaurin | yes, 2 records | fine (but only 2) |
| tolvaptan | yes, 3 records | fine |
| **nitisinone** | **no** | **1** — and it is a **Ki**, not IC50/EC50 |
| **dantrolene** | **no** | **0** |
| **sapropterin** | **no** | **0** |

Two separate causes, two separate fixes:

- **nitisinone** is present but invisible: its only measured activity against CHEMBL1861 is
  `Ki = 37 nM`, and try06 fetches IC50/EC50 only. Fixed by adding a per-target
  `activity_types` override, set to `["IC50", "EC50", "Ki"]` for tyrosinemia alone. This is
  a real change in label semantics for that target — Ki is an equilibrium binding constant,
  not a half-maximal effect concentration — but lower still means tighter, normalization is
  per-target and never compared across targets, and the alternative is no target at all.
  It took tyrosinemia from 68 to 113 ChEMBL records and made the reference findable.
- **Matching itself was fragile.** Fixed by adding `reference_drug_chembl_id` as the primary
  matcher, with `pref_name` as fallback. Confirmed working: the tyrosinemia run reports
  `reference matched by molecule_chembl_id (CHEMBL1337)` — `pref_name` would have aborted
  the target.

---

# mh and pku are reported untrainable, not run

The spec says to report anything under 40 compounds as untrainable rather than run it.
Both qualify on pool size, but they fail something stricter first: **the reference drug has
no measured activity against the target at all**, so there is no anchor to validate any
ranking against. Even with plenty of training data there would be nothing to rank.

**mh (RYR1 / dantrolene)** — three independent disqualifications:
1. Dantrolene (CHEMBL1201288) has **0** activities of any type against CHEMBL1846.
2. CHEMBL1846 has 11 IC50/EC50 records; BindingDB adds 7, PubChem 59, with heavy overlap.
3. RYR1 is **5,038 aa** — 3.4x CFTR, the largest target attempted, and CFTR alone needed
   9h49m for 3 seeds. (Measured: 2,737 prompt tokens, so it would at least *fit* in
   context — the problem is cost, not context.)

**pku (PAH / sapropterin)** — three as well:
1. Sapropterin (CHEMBL1201774) has **0** activities of any type against CHEMBL3076.
2. CHEMBL3076 has exactly **1** IC50/EC50 record. BindingDB 0, PubChem 1.
3. The task's quantity does not apply. Sapropterin **is** tetrahydrobiopterin, PAH's
   natural **cofactor**. This pipeline predicts inhibitory/half-maximal potency; "an
   enzyme's affinity for its own cofactor" is a different measurement, and there is no
   inhibition literature here to learn a structure-activity relationship from. This one
   would not become trainable even if ChEMBL gained thousands of PAH records.

Both are kept in the `TARGETS` dict with an `untrainable` key, so the finding lives in code
and a future session can re-enable either by deleting that key.

---

# Routing outcome

Expectation in the plan was that pku and possibly mh would route to the cross-validation
path. **Neither did** — they are not trainable at all, so there was nothing to route. And
the one target expected to be small, tyrosinemia, routed to **stepstop** rather than
crossval: ChEMBL gave 113 records, but BindingDB (14) and PubChem (68) pushed the
post-dedup pool to **179 distinct compounds**, above the 150 threshold.

Measured for tyrosinemia:

```
raw activities: 113 ChEMBL + 14 BindingDB + 68 PubChem -> 180 unique compounds
reference matched by molecule_chembl_id (CHEMBL1337)
nitisinone real potency = 37.0 nM, normalized score 234
hard negatives: sim 0.63 / 0.62 / 0.47 (the triketone series -- genuine close analogs)
179 distinct compounds -> 895 training lines (5.00x)
ROUTE -> stepstop (179 >= 150)
```

So the crossval path may end up unused in this try. It is wired, tested for syntax, and
reachable — `jobB_small.sb` dispatches on the routing file, not on an assumption — but
unless `cel`/`mastocytosis`/`nsiad` come back smaller than expected, every runnable target
takes the step-stopping path. That is a finding, not a failure: these four orphan targets
have more public binding data than Fabry does.

---

# One more conflict found and fixed

**Imatinib is both the reference drug for `cel` and a member of the fixed `EASY_NEGATIVES`
panel.** Without intervention it would be scored as the compound the model is supposed to
rank highly *and* as an unrelated negative control in the same eval set. Fixed with
`build_easy_negatives`, which filters the panel against the reference drug by **canonical
structure** rather than name, so a synonym cannot slip past. It prints what it dropped.

---

# Prompt token lengths (measured, drives the time limits)

| gene | aa | prompt tokens | |
|---|---|---|---|
| TTR | 147 | 304 | existing |
| AVPR2 | 371 | 415 | new |
| HPD | 393 | 433 | new |
| GLA | 429 | 449 | existing |
| PAH | 452 | 455 | new, untrainable |
| GBA1 | 536 | 501 | existing |
| KIT | 976 | 720 | new |
| PDGFRA | 1089 | 780 | new |
| CFTR | 1480 | 977 | existing, the calibration point |
| RYR1 | 5038 | 2737 | new, untrainable |

Every runnable new target is **at or below CFTR's prompt length**, and CFTR is the most
expensive run this project has completed (9h49m for 3 seeds at 6,050 training lines). That
is what makes the 18h ceiling safe rather than hopeful.
