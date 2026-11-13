# Date-cut retrieval corpus — design

Blinding for an arm that does no training. The other arms remove training compounds
(01) or graph edges (03); this arm's knowledge is in frozen pretrained weights, so the
only thing that *can* be controlled is what it is allowed to retrieve.

**Status: design only. Nothing built. Every date below is a first-pass estimate from
general pharmacology knowledge and MUST be verified against PubMed/DrugBank before
anything is run on it** — see "Verification protocol" at the end. Getting a cutoff wrong
in the permissive direction silently reinstates the leak this whole document exists to
close.

---

## 1. The cutoff is not the approval date

The obvious choice — cut at the drug's approval date — does not work. Approval trails
the discovery literature by years to decades, and it is the *discovery* literature that
contains the answer. Tafamidis was approved in 2011, but it was designed as a TTR
kinetic stabiliser in the Kelly lab and the mechanism was published around 2003. A 2011
cutoff leaves eight years of papers saying exactly what the model is supposed to deduce.

The correct cutoff is:

> **the first publication linking the drug (or its chemical series) to this target or
> disease, minus a safety margin.**

That makes the task genuinely prospective: the agents see the literature as it stood
immediately before the discovery, and must make the discovery.

## 2. The structural problem: not every pin can be date-cut

A cutoff is only coherent if, at that date, **the compound already existed and was
already documented for something else**. Otherwise cutting the corpus doesn't blind the
model — it deletes the candidate from the world, and there is nothing to retrieve about
it at all.

That splits the 12 pins into two genuinely different kinds of case:

- **Repurposing** — the drug was approved and well documented for an unrelated
  indication before anyone connected it to this disease. Date-cutting works perfectly:
  the compound has a rich pre-cutoff literature, and the mechanistic leap is the thing
  being tested.
- **Purpose-designed** — the molecule exists *because* of this disease. Migalastat,
  isofagomine and tafamidis were designed as chaperones/stabilisers for their targets.
  Before the link there is no compound, so there is no honest cutoff.

**This is a real limit on the claim, not a technicality.** RareAgent is a *drug
repurposing* method, and the repurposing subset is where it is defined. The
purpose-designed cases can still be scored, but they test something else — whether the
chemistry tools alone can pick the molecule out — and must be reported separately.

## 3. Proposed cutoffs

Estimates. The "basis" column is what the date is anchored on, and is what verification
must check.

### 3a. Repurposing cases — date-cut is coherent

| disease | drug | cutoff | basis for the date | pre-cutoff status of the drug |
|---|---|---|---|---|
| Gaucher | **ambroxol** | **2008-01-01** | Chaperone activity for glucocerebrosidase reported by Maegawa et al., 2009 | Approved mucolytic since the 1970s, large unrelated literature. **The cleanest case in the set.** |
| Chronic eosinophilic leukemia | **imatinib** | **2002-06-01** | *FIP1L1-PDGFRA* fusion identified as the imatinib-responsive lesion, Cools et al., 2003 | Approved for CML in 2001; BCR-ABL/PDGFR kinase activity well documented |
| Malignant hyperthermia | **dantrolene** | **1974-01-01** | Efficacy in the porcine MH model, Harrison, 1975 | Approved skeletal muscle relaxant from the late 1960s |
| Holocarboxylase synthetase deficiency | **biotin** | **1970-01-01** | Biotin-responsive multiple carboxylase deficiency reported through the 1970s | Vitamin, characterised since the 1940s |
| Propionic acidemia | **biotin** | **1970-01-01** | As above | As above |
| Tyrosinemia type I | **nitisinone** | **1991-01-01** | Lindstedt et al., 1992, first clinical use of NTBC in HT-1 | Existed as an HPPD-inhibitor herbicide chemotype, not as a drug — **borderline**, see note |
| NSIAD | **tolvaptan** | **2009-06-01** | NSIAD described by Feldman et al., 2005; V2R antagonism proposed as treatment subsequently | Approved for hyponatremia in 2009 — **cutoff is tight against approval**, see note |

### 3b. Purpose-designed cases — date-cut is NOT coherent

| disease | drug | why no cutoff works |
|---|---|---|
| Fabry | migalastat | DGJ developed explicitly as a GLA chaperone (Fan et al., 1999). No pre-link existence as a drug. |
| Gaucher | isofagomine | Designed chaperone; never approved, failed Phase 2 around 2009. Already chemistry-arm-only per `pins.json`. |
| ATTR amyloidosis | tafamidis | Designed as a TTR kinetic stabiliser, Kelly lab, ~2003. |
| Phenylketonuria | sapropterin | BH4 *is* PAH's natural cofactor — the "link" is the biochemistry itself and predates any corpus. |
| Mastocytosis | midostaurin | PKC412 was in oncology development; KIT D816V activity reported ~2005, but it was never an approved drug for anything before 2017. Borderline — treat as purpose-designed. |

### Notes on the two borderline rows

- **nitisinone**: NTBC came from herbicide chemistry, so a pre-1991 corpus contains the
  HPPD-inhibition chemistry but no drug. The compound is retrievable as a chemical, not
  as a medicine. Usable, but the finding must say so.
- **tolvaptan**: a 2009 cutoff sits essentially on top of its hyponatremia approval, so
  the pre-cutoff literature for it is thin. If verification shows the NSIAD-tolvaptan
  link was published earlier than assumed, this case moves to 3b.

**Net: 5 clean repurposing cases, 2 borderline, 5 that cannot be date-cut.** That is
enough for a real result, and it is a smaller n than the 12-cell matrix implies. Worth
knowing before the chapter is structured around 12.

## 4. What a cutoff does and does not control

**Controls:** what `literature_search` can return.

**Does not control:** what is already in the base model's weights. Llama-3.3-70B was
trained on text through 2024 and knows that ambroxol is a glucocerebrosidase chaperone.
Cutting the corpus does not unlearn it.

So the corpus cut is **necessary but not sufficient**, and must be paired with:

1. **The familiarity probe** (try13's method) on the chosen base model, reported
   alongside every result.
2. **Blinded disease framing** — `protocol.py` already withholds the disease name and
   the library compound names, passing gene/UniProt/site/mechanism/phenotypes only.
3. **`recall_flag`** — every agent schema in `protocol.py` carries a field asking the
   model to declare when it catches itself recalling a drug-disease pair rather than
   reasoning. Self-report is weak evidence, but a non-empty flag on the winning
   candidate invalidates that case outright.
4. **An anachronism detector** (below) — the one *objective* leak check.

### The anachronism detector

Scan every agent turn for named entities that did not exist before the cutoff: the
reference drug's brand names, the trial identifiers, the gene-fusion name for cases
where that was the discovery, and post-cutoff author/consortium names. Any hit is
positive evidence that parametric memory, not retrieval, produced the answer.

This is cheap, objective, and does not depend on the model's honesty — unlike
`recall_flag`. It should gate every reported number.

## 5. Corpus construction

- **Source**: PubMed baseline + annual update files (title + abstract + MeSH + date).
  Full text is not needed and multiplies the ingest cost.
- **Filter**: keep records with `DateCompleted`/`DateRevised` strictly before the
  case's cutoff. Build **one index per cutoff date**, not one shared index — a shared
  index with runtime filtering will eventually leak through a ranking function that saw
  post-cutoff documents.
- **Retrieval**: BM25 is sufficient and auditable. Dense retrieval needs an embedding
  model, and every current embedding model was itself trained on post-cutoff text —
  a subtler version of the same leak.
- **Scale**: PubMed is ~37M records; the 1970/1974 cutoffs are a few million, the
  2008/2009 cutoffs ~18M. Title+abstract only, this is tens of GB — comfortable against
  the ~765 GB free on `/work/hdd/bisa`.
- **Offline**: compute nodes have no outbound HTTPS (per `discover_candidates.py`'s own
  note). The baseline must be fetched on a login node, like the existing ChEMBL/UniProt
  caches.

## 6. Verification protocol — do this before building anything

For each row in 3a, confirm by direct lookup, and record the PMID in this file:

1. The **earliest** PubMed record linking the drug to the target or disease. The cutoff
   goes *before* it, with a margin of at least 12 months.
2. The drug's approval date and original indication, from DrugBank, to confirm it really
   was documented pre-cutoff for something else.
3. A **negative control search** against the built index: query the index for the
   drug-disease pair and confirm zero relevant hits. If the index returns the answer,
   the cutoff is wrong.

Step 3 is the one that actually proves the blinding, and it is the one most likely to be
skipped. It should be an assertion in the build script, not a manual check.
