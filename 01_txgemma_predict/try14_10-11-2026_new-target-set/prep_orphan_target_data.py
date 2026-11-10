"""
Generalized ChEMBL-based fine-tuning data prep for TxGemma's BindingDB_ic50 task,
for orphan/rare-disease targets where BindingDB itself has little or no coverage
(GLA had 0 BindingDB records of any type; GBA1 had 17).

Lessons encoded here from the Fabry/Gaucher runs:
- A ChEMBL target's real activity data can be split across more than one target
  ChEMBL ID (bulk/"Unchecked" import targets are a known culprit) -- support a
  list of target ChEMBL IDs per disease, not just one.
- The reference/validated drug used to sanity-check a fine-tune must be verified
  to bind the target DIRECTLY, not just be clinically used for the disease.
  Miglustat and eliglustat are real, approved Gaucher drugs, but they work by
  inhibiting UGCG (substrate reduction), not by binding GBA1 -- confirmed via
  ChEMBL activity data and PrimeKG's drug_protein edges. Using them as a
  "the model should rank this well" holdout would have been a mismatched test.
  Ambroxol, by contrast, has real direct GBA1 inhibition/chaperone data
  (IC50 = 4100 nM in a Gaucher-patient fibroblast assay) and is used (via drug
  repurposing / active clinical investigation) specifically for this mechanism.
- "Easy" negative controls (unrelated drug classes -- aspirin, caffeine, ...) are
  a weak test: a model can pass by coarse chemotype recognition alone, without
  any real structure-activity understanding. The meaningful test is "hard"
  negatives: real compounds from the same dataset that are structurally close to
  the reference drug (via RDKit Morgan fingerprint Tanimoto similarity, not
  string heuristics) but empirically much weaker. Example that motivated this:
  a single-stereocenter diastereomer of migalastat scored 992/1000 (near worst
  in the whole GLA set) despite being otherwise identical in 2D structure.
- Some targets will have too little ChEMBL data to fine-tune on at all (this WAS
  true of GLA before ChEMBL rescued it, and will be true of some future orphan
  targets even with ChEMBL). Each target is processed independently so a data
  gap on one target doesn't block the others -- failures are collected and
  reported, not silently skipped.

This script only prepares data (pure CPU, no GPU). Run
lora_finetune_orphan_targets.py separately (GPU) once this has produced
train_blind.jsonl + eval_set.json for each target.
"""
import json
import math
import os
import urllib.error
import urllib.request
from urllib.parse import urlencode

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

RDLogger.DisableLog("rdApp.*")  # pure noise otherwise -- one deprecation warning per fingerprint call


def canon_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None

BASE = "/projects/bisa/sshah21"
OUT_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
# try14 artifacts (routing decision, per-target results, final CSV) live together,
# separate from the per-target data directories so a re-run cannot clobber data.
TRY14_OUT = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan/try14_new_targets"

PROMPT_TMPL = json.load(open(
    "/work/hdd/bisa/sshah21/hf_cache/hub/models--google--txgemma-9b-predict/"
    "snapshots/d8e65c274206c96631276df0ac009a5c6c24b293/tdc_prompts.json"))["BindingDB_ic50"]

# Fixed, chemically-unrelated "easy" negative panel -- cheap sanity floor, not the real test.
EASY_NEGATIVES = {
    "aspirin":   "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine":  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "metformin": "CN(C)C(=N)N=C(N)N",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "imatinib":  "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
}

N_HARD_NEGATIVES = 3
# Below this, fine-tuning is mathematically degenerate (nothing real to learn from) --
# skip. Above it, we still try no matter how thin the data is: this tool exists for
# rare-disease targets, which will often have little data, and the real test of
# whether a small dataset was "enough" is the hard-negative validation step in
# lora_finetune_orphan_targets.py, not a guess made before ever trying. Low-data runs
# are just flagged "low_confidence" in the coverage report so results get extra scrutiny.
MIN_TRAINING_EXAMPLES = 5
LOW_CONFIDENCE_BELOW = 20

# -------------------------------------------------------------------------
# Per-orphan-target configuration. To add a new disease: add an entry here.
#
# chembl_target_ids: find via https://www.ebi.ac.uk/chembl/api/data/target.json
#   ?target_components__accession=<UniProt accession>  -- and check whether the
#   reference drug's real activity data is fragmented onto a second/bulk-import
#   target id (as happened with ambroxol/GBA1 and CHEMBL612545).
#
# reference_drug_name: the ChEMBL molecule_pref_name (lowercased) of a drug you
#   have VERIFIED binds this target directly (real IC50/Ki data against this
#   specific target, ideally cross-checked against a drug_protein edge in
#   PrimeKG if the target has one) -- not just a drug approved for the disease.
# -------------------------------------------------------------------------
TARGETS = {
    "fabry": {
        "fasta": f"{BASE}/GLA.fasta",
        "chembl_target_ids": ["CHEMBL2524"],
        "uniprot": "P06280",
        "reference_drug_name": "migalastat",
    },
    "gaucher": {
        "fasta": f"{BASE}/GBA1.fasta",
        # CHEMBL612545 ("Unchecked") looked useful -- it has 3 real ambroxol/GBA1
        # IC50 values -- but it's actually a generic bulk-import bucket with 266,949
        # total IC50 records spanning many unrelated real targets, not a GBA1-specific
        # assay id. Pulling it whole would inject huge amounts of mislabeled noise for
        # 3 data points already covered by the properly-curated id below. Not worth it.
        "chembl_target_ids": ["CHEMBL2179"],
        "uniprot": "P04062",
        "reference_drug_name": "ambroxol",
    },
    # Two more mechanism classes, deliberately different from GLA/GBA1's substrate-
    # mimicry chaperones -- used for the chat-reasoning leave-one-target-out fine-tune,
    # not the numeric predict task, but the raw data (real reference drug + real
    # hard/easy negatives) is built by the exact same pipeline either way.
    "cf": {
        # Cystic fibrosis / CFTR -- lumacaftor is a FOLDING/TRAFFICKING corrector for
        # the F508del misfolding mutant, not an active-site or substrate mimic at all.
        "fasta": f"{BASE}/CFTR.fasta",
        "chembl_target_ids": ["CHEMBL4051"],
        "uniprot": "P13569",
        "reference_drug_name": "lumacaftor",
    },
    "attr": {
        # Hereditary transthyretin amyloidosis / TTR -- tafamidis kinetically
        # stabilizes the native TTR TETRAMER at a subunit interface, preventing
        # dissociation into amyloid-forming monomers. No catalytic pocket involved.
        "fasta": f"{BASE}/TTR.fasta",
        "chembl_target_ids": ["CHEMBL3194"],
        "uniprot": "P02766",
        "reference_drug_name": "tafamidis",
    },

    # ------------------------------------------------------------------ try14
    # Six new targets. Every ChEMBL id below was vetted by querying
    # target.json?target_components__accession=<UniProt> and printing target_type,
    # organism, pref_name and record counts before being accepted -- the vetting
    # transcript is in VETTING.md. Only SINGLE PROTEIN / Homo sapiens ids were kept;
    # PROTEIN COMPLEX, PROTEIN FAMILY, SELECTIVITY GROUP, CHIMERIC PROTEIN and
    # PROTEIN-PROTEIN INTERACTION ids were rejected. In every case the rejected ids
    # carried either 0 records or a subset already covered by the accepted id.
    #
    # reference_drug_chembl_id is NEW in try14 and is the primary matcher. try06
    # matched the reference drug by exact lowercased molecule_pref_name, which is
    # fragile: it silently fails when ChEMBL stores the drug under a synonym, and a
    # silent failure here aborts the whole target. Matching on the molecule's own
    # ChEMBL id cannot drift. pref_name is kept as a fallback.
    "cel": {
        # Chronic eosinophilic leukemia / PDGFRA. Imatinib inhibits the FIP1L1-PDGFRA
        # fusion kinase -- a genuine direct ATP-competitive inhibitor of this target,
        # not a disease-association guess. 13 imatinib IC50/EC50 records present.
        # NOTE: imatinib is also a member of EASY_NEGATIVES. The panel is filtered
        # against the reference drug at build time (see build_easy_negatives) so it
        # cannot appear as both the thing being ranked and a negative control.
        "fasta": f"{BASE}/PDGFRA.fasta",
        "chembl_target_ids": ["CHEMBL2007"],
        "uniprot": "P16234",
        "reference_drug_name": "imatinib",
        "reference_drug_chembl_id": "CHEMBL941",
    },
    "mastocytosis": {
        # Systemic mastocytosis / KIT. Midostaurin is approved for advanced SM and is a
        # direct multi-kinase inhibitor of KIT D816V. Only 2 midostaurin records, which
        # is enough to anchor the reference but means its own label rests on very few
        # measurements -- recorded here rather than discovered later.
        "fasta": f"{BASE}/KIT.fasta",
        "chembl_target_ids": ["CHEMBL1936"],
        "uniprot": "P10721",
        "reference_drug_name": "midostaurin",
        "reference_drug_chembl_id": "CHEMBL608533",
    },
    "nsiad": {
        # Nephrogenic syndrome of inappropriate antidiuresis / AVPR2. Tolvaptan is a
        # selective V2 receptor ANTAGONIST -- a direct competitive binder, and the
        # mechanism is receptor blockade rather than chaperoning, which makes this a
        # deliberately different mechanism class from the four existing targets.
        "fasta": f"{BASE}/AVPR2.fasta",
        "chembl_target_ids": ["CHEMBL1790"],
        "uniprot": "P30518",
        "reference_drug_name": "tolvaptan",
        "reference_drug_chembl_id": "CHEMBL344159",
    },
    "tyrosinemia": {
        # Tyrosinemia type I / HPD (HPPD). Nitisinone is a direct HPPD inhibitor.
        #
        # THE HERBICIDE TRAP IS ALREADY AVOIDED, AND NOT BY AN ORGANISM FILTER.
        # The large HPPD literature is against PLANT HPPD, but those assays live under
        # DIFFERENT ChEMBL target ids whose organism is a plant species. Querying by the
        # HUMAN UniProt accession P32754 returns exactly one target, CHEMBL1861
        # (SINGLE PROTEIN / Homo sapiens, 68 IC50 records), so the plant data is never
        # fetched in the first place. Spot-checking the assay descriptions confirms it:
        # "Inhibition of human recombinant His-tagged HPPD expressed in Escherichia
        # coli BL21(DE3)" -- human enzyme, bacterial expression host, which is normal
        # and correct. See VETTING.md for why the assay_organism filter that was asked
        # for would have deleted the entire dataset instead of protecting it.
        #
        # Ki IS REQUIRED FOR THIS TARGET. Nitisinone's only measured activity against
        # CHEMBL1861 is a Ki of 37 nM; it has ZERO IC50/EC50 records. With try06's
        # IC50/EC50-only fetch the reference drug is unfindable and the target aborts.
        # Ki is a true equilibrium binding constant rather than a half-maximal effect
        # concentration, so this is a real change in label semantics for this target --
        # but lower still means tighter, the normalization is per-target and is never
        # compared across targets, and the alternative is no target at all.
        "fasta": f"{BASE}/HPD.fasta",
        "chembl_target_ids": ["CHEMBL1861"],
        "uniprot": "P32754",
        "reference_drug_name": "nitisinone",
        "reference_drug_chembl_id": "CHEMBL1337",
        "activity_types": ["IC50", "EC50", "Ki"],
    },
    "mh": {
        # Malignant hyperthermia / RYR1. REPORTED UNTRAINABLE -- see VETTING.md.
        # Kept in the dict so the finding is recorded in code rather than only in prose,
        # and so a future session with better data can flip it on by deleting
        # "untrainable". Three independent disqualifications, any one of which is fatal:
        #   1. Dantrolene (CHEMBL1201288) has ZERO activities of ANY type against
        #      CHEMBL1846. There is no reference anchor, so even with data there would
        #      be nothing to validate the ranking against.
        #   2. CHEMBL1846 has 11 IC50/EC50 records total; BindingDB adds 7 and PubChem
        #      59, and those overlap heavily. Far below the 40-compound floor.
        #   3. RYR1 is 5,038 aa -- 3.4x CFTR, the largest target attempted so far, and
        #      CFTR alone needed ~10h for 3 seeds.
        "fasta": f"{BASE}/RYR1.fasta",
        "chembl_target_ids": ["CHEMBL1846"],
        "uniprot": "P21817",
        "reference_drug_name": "dantrolene",
        "reference_drug_chembl_id": "CHEMBL1201288",
        "untrainable": "dantrolene has 0 measured activities of any type against "
                       "CHEMBL1846 (no reference anchor); 11 ChEMBL IC50/EC50 records "
                       "total; RYR1 is 5038 aa (3.4x CFTR)",
    },
    "pku": {
        # Phenylketonuria / PAH. REPORTED UNTRAINABLE -- see VETTING.md.
        #   1. Sapropterin (CHEMBL1201774) has ZERO activities of ANY type against
        #      CHEMBL3076. No reference anchor.
        #   2. CHEMBL3076 has exactly 1 IC50/EC50 record; BindingDB has 0, PubChem 1.
        #   3. Mechanistically this target does not fit the task even with data:
        #      sapropterin IS tetrahydrobiopterin, PAH's natural COFACTOR. The pipeline
        #      predicts inhibitory/half-maximal potency; "affinity of an enzyme for its
        #      own cofactor" is a different quantity, and there is no inhibition
        #      literature here to learn a structure-activity relationship from.
        "fasta": f"{BASE}/PAH.fasta",
        "chembl_target_ids": ["CHEMBL3076"],
        "uniprot": "P00439",
        "reference_drug_name": "sapropterin",
        "reference_drug_chembl_id": "CHEMBL1201774",
        "untrainable": "sapropterin has 0 measured activities of any type against "
                       "CHEMBL3076 (no reference anchor); 1 ChEMBL IC50/EC50 record "
                       "total; and sapropterin is PAH's natural cofactor, not an "
                       "inhibitor, so the task's quantity does not apply",
    },
}

# Routing thresholds (try14). Below UNTRAINABLE_BELOW a target is reported and skipped
# rather than run; between that and CROSSVAL_BELOW it takes try12's 6-fold scaffold CV
# path; at or above it takes try11's step-level-stopping path.
#
# Why 150: Fabry's 89 compounds left 11 validation molecules after the 15% split and the
# scaffold holdout, which made checkpoint selection close to arbitrary (try11 spread
# 38.4). Cross-validating put every compound in validation exactly once and collapsed
# the spread to 4.6 (try12). 150 compounds is roughly where a 15% split still leaves
# ~22 validation molecules, which is the point at which a single split starts to be
# worth measuring at all.
UNTRAINABLE_BELOW = 40
CROSSVAL_BELOW = 150

# Runtime cap on training compounds (try14). KIT has ~4,350 ChEMBL IC50 records plus
# ~6,175 usable PubChem rows; unbounded that is several times CF's 1,450 compounds, and
# CF alone took 9h49m for 3 seeds. An unattended chain cannot absorb that, and try07
# already established that volume is not the lever -- Gaucher passed at 493 compounds
# after downsampling from 1,208. So large targets are subsampled to a MaxMin-DIVERSE
# subset rather than truncated arbitrarily: diversity selection keeps the chemical space
# spanned, which is what matters, instead of keeping whichever compounds ChEMBL happened
# to return first. Both the raw and capped counts are recorded.
MAX_TRAIN_COMPOUNDS = int(os.environ.get("MAX_TRAIN_COMPOUNDS", "1500"))


def read_fasta(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


MAX_CHEMBL_RECORDS = 5000  # a real single-protein target's curated IC50 set doesn't
# approach this. Bulk/"Unchecked" import target ids (a real ChEMBL data quirk -- see
# CHEMBL612545 above, 266,949 records spanning many unrelated targets) blow past it
# by two orders of magnitude. Refuse rather than silently ingest probable noise.


ACTIVITY_TYPES = ["IC50", "EC50"]  # IC50 for inhibitors, EC50 for activators/correctors
# (e.g. CFTR correctors like lumacaftor have no IC50 at all -- they don't inhibit
# anything, they correct folding, so potency is only ever reported as EC50). Both are
# "concentration for half-maximal effect" -- lower is more potent either way, so mixing
# them within one target's min-max normalization is valid as long as we don't compare
# raw values ACROSS targets, which we never do.


# Cache of assay_chembl_id -> assay_organism, so the organism guard costs one request
# per distinct ASSAY rather than one per activity record (tyrosinemia: 68 activities but
# only a handful of distinct assays).
_ASSAY_ORGANISM_CACHE = {}


def assay_organism(assay_chembl_id):
    if assay_chembl_id in _ASSAY_ORGANISM_CACHE:
        return _ASSAY_ORGANISM_CACHE[assay_chembl_id]
    org = None
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/assay/{assay_chembl_id}.json"
        with urllib.request.urlopen(url, timeout=30) as r:
            org = json.load(r).get("assay_organism")
    except Exception:
        org = None
    _ASSAY_ORGANISM_CACHE[assay_chembl_id] = org
    return org


def fetch_chembl_activities(target_chembl_id, activity_types=None,
                            require_human_assay=False):
    """Returns list of (smiles, log10(nM), molecule_name, molecule_chembl_id) from
    ChEMBL, across activity_types (defaults to the module-level ACTIVITY_TYPES).

    require_human_assay drops only records whose assay organism is EXPLICITLY a
    non-human species. Records with no organism recorded are KEPT -- see VETTING.md:
    the activity endpoint returns assay_organism = null for every record on all six
    try14 targets, so dropping nulls would delete 100% of the data on every one of
    them. The real protection against cross-species contamination is choosing a
    SINGLE PROTEIN / Homo sapiens target id, which is done at config time.
    """
    out = []
    for standard_type in (activity_types or ACTIVITY_TYPES):
        peek_url = ("https://www.ebi.ac.uk/chembl/api/data/activity.json?" +
                    urlencode({"target_chembl_id": target_chembl_id, "standard_type": standard_type,
                               "standard_units": "nM", "limit": 1}))
        with urllib.request.urlopen(peek_url, timeout=30) as r:
            total = json.load(r)["page_meta"]["total_count"]
        if total > MAX_CHEMBL_RECORDS:
            raise ValueError(f"{target_chembl_id} has {total} {standard_type} records -- that's "
                              f"almost certainly a bulk/multi-target import id, not a real "
                              f"single-protein assay target. Refusing to pull it; find the "
                              f"correctly-curated id instead of accepting whatever's tagged here.")

        activities = []
        url = ("https://www.ebi.ac.uk/chembl/api/data/activity.json?" +
               urlencode({"target_chembl_id": target_chembl_id, "standard_type": standard_type,
                          "standard_units": "nM", "limit": 1000}))
        while url:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
            activities.extend(data["activities"])
            url = data["page_meta"]["next"]
            if url:
                url = "https://www.ebi.ac.uk" + url

        n_nonhuman = 0
        for a in activities:
            smi, val = a.get("canonical_smiles"), a.get("standard_value")
            if not smi or val is None:
                continue
            try:
                val = float(val)
            except ValueError:
                continue
            if val <= 0:
                continue
            if require_human_assay and a.get("assay_chembl_id"):
                org = a.get("assay_organism") or assay_organism(a["assay_chembl_id"])
                if org is not None and org != "Homo sapiens":
                    n_nonhuman += 1
                    continue
            out.append((smi, math.log10(val),
                        (a.get("molecule_pref_name") or "").strip().lower(),
                        a.get("molecule_chembl_id") or ""))
        if require_human_assay and n_nonhuman:
            print(f"  organism guard: dropped {n_nonhuman} {standard_type} record(s) with an "
                  f"explicitly non-human assay organism")
    return out


def fetch_bindingdb_activities(uniprot_id):
    """Returns list of (smiles, log10(nM), '') from BindingDB, across ACTIVITY_TYPES.
    Only exact-valued entries (skips '<'/'>'-qualified values -- exact potency unknown)."""
    url = f"https://bindingdb.org/rest/getLigandsByUniprot?uniprot={uniprot_id}&response=application/json"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    affinities = data.get("getLindsByUniprotResponse", {}).get("bdb.affinities", [])

    out = []
    for a in affinities:
        if a.get("bdb.affinity_type") not in ACTIVITY_TYPES:
            continue
        smi = a.get("bdb.smile", "").replace(" |r|", "")
        raw = str(a.get("bdb.affinity", "")).strip()
        if not smi or raw.startswith("<") or raw.startswith(">"):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if val <= 0:
            continue
        out.append((smi, math.log10(val), "", ""))
    return out


def fetch_pubchem_activities(uniprot_accession):
    """Returns list of (smiles, log10(IC50_nM), '') from PubChem BioAssay (PCBA).

    PCBA is much larger than ChEMBL/BindingDB (tens of thousands of rows for a
    well-screened target) but noisier: most rows are single-concentration primary
    screening hits with no real potency number, and its target-gene tagging is
    less carefully curated than ChEMBL's (spot-checked example: a Ki record tagged
    to human GBA's gene ID was actually an Agrobacterium sp. ortholog assay, not
    the human enzyme). To keep this usable, we only take rows that: (1) report a
    real IC50 value, not a bare Active/Inactive flag, and (2) come from a
    Confirmatory or Screening dose-response assay, not a bulk primary screen.
    Even so, treat this as a noisier supplementary source, not as trustworthy as
    ChEMBL row-for-row.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/protein/accession/{uniprot_accession}/concise/JSON"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError:
        return []
    if "Fault" in data:
        return []

    rows = data.get("Table", {}).get("Row", [])
    cols = data.get("Table", {}).get("Columns", {}).get("Column", [])

    cid_to_logvals = {}
    for row in rows:
        c = dict(zip(cols, row["Cell"]))
        if c.get("Activity Name") not in ACTIVITY_TYPES or c.get("Assay Type") not in ("Confirmatory", "Screening"):
            continue
        val, cid = c.get("Activity Value [uM]"), c.get("CID")
        if not val or not cid:
            continue
        try:
            val_nm = float(val) * 1000  # PCBA reports uM, we normalize in nM throughout
        except ValueError:
            continue
        if val_nm <= 0:
            continue
        cid_to_logvals.setdefault(cid, []).append(math.log10(val_nm))

    if not cid_to_logvals:
        return []

    cid_to_smiles = {}
    cids = list(cid_to_logvals.keys())
    for i in range(0, len(cids), 150):
        batch = cids[i:i + 150]
        prop_url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" +
                    ",".join(batch) + "/property/CanonicalSMILES/JSON")
        try:
            with urllib.request.urlopen(prop_url, timeout=30) as r:
                pdata = json.load(r)
        except Exception:
            continue
        for prop in pdata.get("PropertyTable", {}).get("Properties", []):
            smi = prop.get("CanonicalSMILES") or prop.get("ConnectivitySMILES")
            if smi:
                cid_to_smiles[str(prop["CID"])] = smi

    return [(cid_to_smiles[cid], sum(v) / len(v), "", "")
            for cid, v in cid_to_logvals.items() if cid in cid_to_smiles]


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def make_prompt(smiles, sequence):
    return PROMPT_TMPL.replace("{Drug SMILES}", smiles).replace("{Target amino acid sequence}", sequence)


N_SMILES_AUGMENT = 4  # extra randomized-but-equivalent SMILES per training compound,
# on top of its canonical form. A SMILES string isn't unique -- the same molecule can
# be written many valid ways -- and training on only the canonical form let a previous
# fine-tune memorize exact strings instead of learning real structure-activity signal
# (diagnosed from evidence: predicted scores exactly matched the real training labels
# of near-identical-text neighbors, not the queried compound's own chemistry). This is
# a fix for the TRAINING pool only -- eval_set.json (reference/hard/easy negatives)
# stays as single canonical SMILES since those are scored once, not trained on.


def augment_smiles(smiles, n=N_SMILES_AUGMENT):
    """[canonical, random_variant, ...] -- up to n+1 distinct SMILES for one molecule.
    Returns fewer if RDKit can't produce that many distinct random traversals (small
    or highly symmetric molecules)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    variants = {smiles}
    for _ in range(n * 3):  # a few extra draws since random traversals can repeat
        if len(variants) > n:
            break
        variants.add(Chem.MolToSmiles(mol, doRandom=True))
    return list(variants)


def safe_fetch(label, fn, *args):
    """A network hiccup on one source should cost us that source, not the whole run."""
    try:
        return fn(*args)
    except Exception as e:
        print(f"  WARNING: {label} fetch failed ({type(e).__name__}: {e}) -- treating as 0 records.")
        return []


def cap_pool_by_diversity(name, pool, cap):
    """Subsample pool to `cap` compounds, MaxMin-DIVERSE rather than arbitrary.

    Truncating ChEMBL's return order would bias the training set toward whichever
    chemotypes happen to be paginated first. MaxMin picks the subset that spans the
    widest chemical space, which is the property that matters -- and try07 established
    that volume is not the lever anyway (Gaucher passed at 493 compounds, down from
    1,208). Returns the pool unchanged when it is already under the cap.
    """
    if cap <= 0 or len(pool) <= cap:
        return pool, False
    mols = [(r, Chem.MolFromSmiles(r["smiles"])) for r in pool]
    mols = [(r, m) for r, m in mols if m is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for _, m in mols]
    idx = list(MaxMinPicker().LazyBitVectorPick(fps, len(fps), cap, seed=0))
    kept = [mols[i][0] for i in idx]
    print(f"[{name}] pool capped {len(pool)} -> {len(kept)} compounds by MaxMin diversity "
          f"(MAX_TRAIN_COMPOUNDS={cap}); chemical space spanned, not truncated")
    return kept, True


def build_easy_negatives(reference):
    """EASY_NEGATIVES minus the reference drug itself.

    Needed from try14 on: imatinib is both a member of the fixed easy-negative panel
    and the reference drug for `cel`. Without this filter imatinib would be scored as
    the thing the model is supposed to rank highly AND as an unrelated negative
    control in the same eval set, which is incoherent. Matched on canonical structure,
    not name, so a synonym cannot slip through.
    """
    ref_canon = canon_smiles(reference["smiles"])
    out, dropped = {}, []
    for n, smi in EASY_NEGATIVES.items():
        if canon_smiles(smi) == ref_canon:
            dropped.append(n)
            continue
        out[n] = smi
    if dropped:
        print(f"  easy-negative panel: dropped {dropped} -- same compound as the reference drug")
    return out


def build_target_dataset(name, cfg):
    records = []  # (smiles, log10_nM, molecule_name, molecule_chembl_id)
    n_chembl = 0
    act_types = cfg.get("activity_types")
    if act_types:
        print(f"[{name}] activity types for this target: {act_types} "
              f"(module default is {ACTIVITY_TYPES})")
    for tid in cfg["chembl_target_ids"]:
        r = safe_fetch("ChEMBL", fetch_chembl_activities, tid, act_types,
                       cfg.get("require_human_assay", False))
        records.extend(r)
        n_chembl += len(r)

    n_bindingdb = n_pubchem = 0
    if cfg.get("uniprot"):
        r = safe_fetch("BindingDB", fetch_bindingdb_activities, cfg["uniprot"])
        records.extend(r)
        n_bindingdb = len(r)

        r = safe_fetch("PubChem BioAssay", fetch_pubchem_activities, cfg["uniprot"])
        records.extend(r)
        n_pubchem = len(r)

    # dedup across ALL sources by canonical structure, not raw SMILES string --
    # different databases write the same molecule differently often enough that a
    # plain string comparison undercounts real overlap.
    by_smiles, names_by_smiles, ids_by_smiles = {}, {}, {}
    n_unparseable = 0
    for smi, logval, mol_name, mol_id in records:
        csmi = canon_smiles(smi)
        if csmi is None:
            n_unparseable += 1
            continue
        by_smiles.setdefault(csmi, []).append(logval)
        if mol_name:
            names_by_smiles[csmi] = mol_name
        if mol_id:
            ids_by_smiles[csmi] = mol_id

    print(f"[{name}] raw activities: {n_chembl} ChEMBL + {n_bindingdb} BindingDB + "
          f"{n_pubchem} PubChem BioAssay -> {len(by_smiles)} unique compounds after "
          f"cross-source dedup ({n_unparseable} unparseable SMILES dropped)")

    log_vals = {smi: sum(v) / len(v) for smi, v in by_smiles.items()}
    if len(log_vals) < MIN_TRAINING_EXAMPLES:
        raise ValueError(f"only {len(log_vals)} usable IC50 compounds across ChEMBL "
                          f"{cfg['chembl_target_ids']} + BindingDB/PubChem for {cfg.get('uniprot')} "
                          f"(need >= {MIN_TRAINING_EXAMPLES}) -- none of the three sources this "
                          f"script supports have enough data for this target.")

    lo, hi = min(log_vals.values()), max(log_vals.values())

    def normalize(v):
        return 500 if hi == lo else round(1000 * (v - lo) / (hi - lo))

    # Reference resolution, try14: ChEMBL MOLECULE ID first, pref_name only as fallback.
    # try06 matched on lowercased molecule_pref_name alone. That is fragile -- ChEMBL
    # stores plenty of drugs under a synonym or with pref_name unset on individual
    # activity rows -- and a miss aborts the entire target with a confusing message. The
    # molecule's own ChEMBL id cannot drift, so it is the primary matcher wherever the
    # config supplies one.
    ref_name = cfg["reference_drug_name"].lower()
    ref_id = cfg.get("reference_drug_chembl_id")
    reference, pool, matched_by = None, [], None
    for smi, logv in log_vals.items():
        rec = {"smiles": smi, "name": names_by_smiles.get(smi, ""), "score": normalize(logv),
               "potency_nM": round(10 ** logv, 1),
               "molecule_chembl_id": ids_by_smiles.get(smi, "")}
        is_ref = False
        if ref_id and rec["molecule_chembl_id"] == ref_id:
            is_ref, matched_by = True, "molecule_chembl_id"
        elif reference is None and rec["name"] == ref_name:
            is_ref, matched_by = True, "molecule_pref_name"
        if is_ref and reference is None:
            reference = rec
        else:
            pool.append(rec)

    if reference is None:
        raise ValueError(f"reference drug '{cfg['reference_drug_name']}' "
                          f"(id {ref_id or 'not configured'}) not found among "
                          f"{len(log_vals)} compounds for {cfg['chembl_target_ids']} -- it has "
                          f"no measured activity of the fetched types against this target, so "
                          f"there is no anchor to validate a ranking against. Verify the id, or "
                          f"whether this target needs additional activity_types (nitisinone "
                          f"needed Ki), before trusting this config.")
    print(f"[{name}] reference matched by {matched_by}"
          + (f" ({reference['molecule_chembl_id']})" if reference["molecule_chembl_id"] else ""))

    # Different sources sometimes record the SAME real compound with vs. without
    # explicit stereochemistry (spotted for ambroxol/GBA1: one source's entry had no
    # defined stereocenters at all, so canonical-SMILES dedup treated it as a
    # "different" molecule from the reference even though it's the same real
    # substance). That's a duplicate to drop -- but a compound sharing the reference's
    # flat skeleton with its OWN fully-defined-but-different stereochemistry is a
    # genuine diastereomer: a real, chemically distinct compound (this is exactly the
    # migalastat-diastereomer case that motivated hard negatives in the first place --
    # scored 992/1000 despite an otherwise-identical 2D structure). Only drop the
    # former: same skeleton AND stereocenters left unassigned, not just "differs from
    # the reference."
    def skeleton_and_defined_stereocenters(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, 0
        skeleton = Chem.MolToSmiles(mol, isomericSmiles=False)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        n_defined = sum(1 for _, tag in centers if tag != '?')
        return skeleton, n_defined

    ref_skeleton, ref_n_defined = skeleton_and_defined_stereocenters(reference["smiles"])

    def is_annotation_gap_duplicate(rec):
        skeleton, n_defined = skeleton_and_defined_stereocenters(rec["smiles"])
        return skeleton == ref_skeleton and ref_n_defined > 0 and n_defined == 0

    before = len(pool)
    pool = [r for r in pool if not is_annotation_gap_duplicate(r)]
    if before != len(pool):
        print(f"[{name}] dropped {before - len(pool)} pool compound(s) that are the reference "
              f"itself under different stereochemistry annotation, not real distinct compounds.")

    # A "PASS" later just means the model ranked this reference above the negatives --
    # it says nothing about whether the reference is actually a potent binder in
    # absolute terms. Print the real potency (IC50 or EC50, whichever this compound was measured as) so that's never mistaken for the same thing
    # (e.g. ambroxol's real potency against GBA1 is 4100 nM -- a weak/modest number
    # compared to migalastat's 13-70 nM against GLA -- it's just the best-documented
    # direct binder we could find, not a strong one).
    print(f"[{name}] reference '{reference['name']}' real potency = {reference['potency_nM']} nM "
          f"(this is an absolute potency number, not a comparison to other targets -- judge "
          f"it on its own terms, don't assume 'passed validation' means 'is a strong binder').")

    # Hard negatives: the pool compounds most structurally similar to the reference
    # drug (real near-miss analogs), regardless of their score -- these are the ones
    # that actually test whether the model learned the right thing.
    ref_fp = fingerprint(reference["smiles"])
    scored = []
    for rec in pool:
        fp = fingerprint(rec["smiles"])
        if fp is None or ref_fp is None:
            continue
        sim = DataStructs.TanimotoSimilarity(ref_fp, fp)
        scored.append((sim, rec))
    scored.sort(key=lambda x: -x[0])
    hard_negatives = [rec for sim, rec in scored[:N_HARD_NEGATIVES]]

    print(f"[{name}] {len(pool) + 1} total usable compounds "
          f"({len(cfg['chembl_target_ids'])} ChEMBL target id(s)). "
          f"Reference '{reference['name']}' real score={reference['score']} (held out of training).")
    print(f"[{name}] hard negatives (Tanimoto similarity to reference):")
    for sim, r in scored[:N_HARD_NEGATIVES]:
        print(f"    {r['name'] or r['smiles'][:30]}  sim={sim:.2f}  real score={r['score']}")

    # Cap AFTER hard negatives are chosen, so the diversity subsample can never silently
    # remove the three compounds the hard-negative test depends on. They are re-inserted
    # if the picker dropped them -- the eval set must stay fixed regardless of the cap.
    n_raw = len(pool)
    pool, was_capped = cap_pool_by_diversity(name, pool, MAX_TRAIN_COMPOUNDS)
    if was_capped:
        kept = {r["smiles"] for r in pool}
        readded = [h for h in hard_negatives if h["smiles"] not in kept]
        if readded:
            pool.extend(readded)
            print(f"[{name}] re-added {len(readded)} hard negative(s) the diversity picker "
                  f"dropped -- the eval set is fixed and must not move with the cap")

    return pool, reference, hard_negatives, n_raw


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Scope to just the targets this run is actually for (TARGETS_TO_RUN env var,
    # comma-separated -- e.g. "cf,attr"). Without this, prep silently processes every
    # target in the shared TARGETS dict, and a later finetune run reading
    # coverage_report.json has no way to tell "ready this run" from "ready from a
    # previous run" -- the exact coupling bug that caused try05's job to also
    # (unintentionally) attempt cf/attr and time out. Merge into the existing
    # coverage_report.json rather than overwrite it, so an already-covered target
    # (e.g. fabry/gaucher) isn't wiped out by a run that isn't touching it.
    _scope = os.environ.get("TARGETS_TO_RUN")
    run_targets = {k: v for k, v in TARGETS.items() if k in _scope.split(",")} if _scope else TARGETS
    print(f"Running prep for: {list(run_targets.keys())}"
          + (" (scoped via TARGETS_TO_RUN)" if _scope else " (no TARGETS_TO_RUN set -- all targets)"))

    cov_path = os.path.join(OUT_DIR, "coverage_report.json")
    coverage_report = json.load(open(cov_path)) if os.path.exists(cov_path) else {}

    routing = {}

    for name, cfg in run_targets.items():
        # Targets disqualified at config time are reported, not run. The spec's floor is
        # a pool-size floor, but these two fail on something stricter: the reference drug
        # has NO measured activity against the target at all, so there is no anchor to
        # validate any ranking against. Running them would burn GPU hours to produce a
        # number with nothing to compare it to.
        if cfg.get("untrainable"):
            print(f"[{name}] UNTRAINABLE -- {cfg['untrainable']}")
            coverage_report[name] = {"status": "untrainable", "reason": cfg["untrainable"]}
            routing[name] = {"path": "none", "reason": cfg["untrainable"],
                             "n_training": 0}
            continue
        try:
            seq = read_fasta(cfg["fasta"])
            pool, reference, hard_negatives, n_raw = build_target_dataset(name, cfg)
        except ValueError as e:
            print(f"[{name}] SKIPPED -- {e}")
            coverage_report[name] = {"status": "skipped", "reason": str(e)}
            routing[name] = {"path": "none", "reason": str(e), "n_training": 0}
            continue

        result_dir = os.path.join(OUT_DIR, name)
        os.makedirs(result_dir, exist_ok=True)

        n_written = 0
        with open(os.path.join(result_dir, "train_blind.jsonl"), "w") as f:
            for rec in pool:
                for smi_variant in augment_smiles(rec["smiles"]):
                    f.write(json.dumps({"prompt": make_prompt(smi_variant, seq),
                                         "answer": str(rec["score"])}) + "\n")
                    n_written += 1
        print(f"[{name}] {len(pool)} distinct compounds -> {n_written} training lines "
              f"after SMILES augmentation ({n_written / len(pool):.1f}x)")

        eval_set = {
            "reference": {**reference, "prompt": make_prompt(reference["smiles"], seq)},
            "hard_negatives": [{**r, "prompt": make_prompt(r["smiles"], seq)} for r in hard_negatives],
            "easy_negatives": [{"name": n, "smiles": sm, "prompt": make_prompt(sm, seq)}
                                for n, sm in build_easy_negatives(reference).items()],
        }
        with open(os.path.join(result_dir, "eval_set.json"), "w") as f:
            json.dump(eval_set, f, indent=2)

        # ---- ROUTING. The whole point of job A: decide the eval path from the measured
        # pool size, so job B's array can run unattended with the right harness per target.
        n_training = len(pool)
        if n_training < UNTRAINABLE_BELOW:
            path = "none"
            reason = (f"{n_training} distinct compounds is below the {UNTRAINABLE_BELOW}-compound "
                      f"floor; a scaffold holdout plus any validation split would leave "
                      f"essentially nothing to measure on")
            status = "untrainable"
            print(f"[{name}] UNTRAINABLE -- {reason}")
        elif n_training < CROSSVAL_BELOW:
            path = "crossval"
            reason = (f"{n_training} < {CROSSVAL_BELOW}: a single 15% split would leave "
                      f"~{max(1, int(0.15 * n_training))} validation molecules, which is the "
                      f"regime where try11's Fabry selection was arbitrary (spread 38.4) and "
                      f"try12's 6-fold CV fixed it (spread 4.6)")
            status = "ready_crossval"
            print(f"[{name}] ROUTE -> crossval ({reason})")
        else:
            path = "stepstop"
            reason = (f"{n_training} >= {CROSSVAL_BELOW}: a 15% split leaves "
                      f"~{int(0.15 * n_training)} validation molecules, enough for try11's "
                      f"step-level stopping to select on")
            status = "ready"
            print(f"[{name}] ROUTE -> stepstop ({reason})")

        seq_len = len(seq)
        routing[name] = {"path": path, "reason": reason, "n_training": n_training,
                         "n_training_raw_before_cap": n_raw,
                         "n_training_lines": n_written,
                         "augmentation_factor": round(n_written / max(1, len(pool)), 2),
                         "sequence_length_aa": seq_len,
                         "reference_drug": reference["name"] or cfg["reference_drug_name"],
                         "reference_real_score": reference["score"],
                         "reference_potency_nM": reference["potency_nM"],
                         "activity_types": cfg.get("activity_types") or ACTIVITY_TYPES}
        if status == "low_confidence":
            print(f"[{name}] WARNING: only {len(pool)} distinct training compounds.")
        print(f"[{name}] wrote eval_set.json to {result_dir}\n")
        coverage_report[name] = {"status": status, "n_training": n_training,
                                 "n_training_lines": n_written, "path": path}

    with open(os.path.join(OUT_DIR, "coverage_report.json"), "w") as f:
        json.dump(coverage_report, f, indent=2)

    # The routing file is job B's input: its array reads this to pick a harness per target.
    routing_path = os.path.join(TRY14_OUT, "routing.json")
    os.makedirs(TRY14_OUT, exist_ok=True)
    with open(routing_path, "w") as f:
        json.dump(routing, f, indent=2)

    print("=== POOL SIZES AND ROUTING ===")
    print(f"{'target':14s} {'path':9s} {'n_train':>8s} {'raw':>7s} {'lines':>7s} "
          f"{'aug':>5s} {'seq_aa':>7s} {'ref':>14s} {'ref_score':>9s}")
    for name, r in routing.items():
        if r["path"] == "none":
            print(f"{name:14s} {'NONE':9s} {r.get('n_training', 0):>8} "
                  f"{'-':>7s} {'-':>7s} {'-':>5s} {'-':>7s} {'-':>14s} {'-':>9s}")
            continue
        print(f"{name:14s} {r['path']:9s} {r['n_training']:>8} "
              f"{r['n_training_raw_before_cap']:>7} {r['n_training_lines']:>7} "
              f"{r['augmentation_factor']:>5.2f} {r['sequence_length_aa']:>7} "
              f"{r['reference_drug'][:14]:>14s} {r['reference_real_score']:>9}")

    runnable = [n for n, r in routing.items() if r["path"] != "none"]
    print(f"\nrunnable targets for job B: {runnable}")
    print(f"  stepstop: {[n for n, r in routing.items() if r['path'] == 'stepstop']}")
    print(f"  crossval: {[n for n, r in routing.items() if r['path'] == 'crossval']}")
    print(f"  NOT RUN:  {[n for n, r in routing.items() if r['path'] == 'none']}")
    print(f"\nrouting written to {routing_path}")
    for name, r in routing.items():
        if r["path"] == "none":
            print(f"\n{name}: NOT RUN -- {r['reason']}")
