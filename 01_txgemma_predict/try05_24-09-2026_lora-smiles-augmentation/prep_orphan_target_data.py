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

RDLogger.DisableLog("rdApp.*")  # pure noise otherwise -- one deprecation warning per fingerprint call


def canon_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None

BASE = "/projects/bisa/sshah21"
OUT_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"

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
}


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


def fetch_chembl_activities(target_chembl_id):
    """Returns list of (smiles, log10(nM), molecule_name) from ChEMBL, across
    ACTIVITY_TYPES."""
    out = []
    for standard_type in ACTIVITY_TYPES:
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
            out.append((smi, math.log10(val), (a.get("molecule_pref_name") or "").strip().lower()))
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
        out.append((smi, math.log10(val), ""))
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

    return [(cid_to_smiles[cid], sum(v) / len(v), "")
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


def build_target_dataset(name, cfg):
    records = []  # (smiles, log10_ic50_nM, name)
    n_chembl = 0
    for tid in cfg["chembl_target_ids"]:
        r = safe_fetch("ChEMBL", fetch_chembl_activities, tid)
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
    by_smiles, names_by_smiles = {}, {}
    n_unparseable = 0
    for smi, logval, mol_name in records:
        csmi = canon_smiles(smi)
        if csmi is None:
            n_unparseable += 1
            continue
        by_smiles.setdefault(csmi, []).append(logval)
        if mol_name:
            names_by_smiles[csmi] = mol_name

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

    ref_name = cfg["reference_drug_name"].lower()
    reference, pool = None, []
    for smi, logv in log_vals.items():
        rec = {"smiles": smi, "name": names_by_smiles.get(smi, ""), "score": normalize(logv),
               "potency_nM": round(10 ** logv, 1)}
        if rec["name"] == ref_name:
            reference = rec
        else:
            pool.append(rec)

    if reference is None:
        raise ValueError(f"reference drug '{cfg['reference_drug_name']}' not found by exact "
                          f"molecule_pref_name match among {len(log_vals)} ChEMBL compounds for "
                          f"{cfg['chembl_target_ids']} -- verify spelling, or that it's really "
                          f"annotated against this target id, before trusting this config.")

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

    return pool, reference, hard_negatives


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    coverage_report = {}

    for name, cfg in TARGETS.items():
        try:
            seq = read_fasta(cfg["fasta"])
            pool, reference, hard_negatives = build_target_dataset(name, cfg)
        except ValueError as e:
            print(f"[{name}] SKIPPED -- {e}")
            coverage_report[name] = {"status": "skipped", "reason": str(e)}
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
            "easy_negatives": [{"name": n, "smiles": s, "prompt": make_prompt(s, seq)}
                                for n, s in EASY_NEGATIVES.items()],
        }
        with open(os.path.join(result_dir, "eval_set.json"), "w") as f:
            json.dump(eval_set, f, indent=2)

        status = "ready" if len(pool) >= LOW_CONFIDENCE_BELOW else "low_confidence"
        if status == "low_confidence":
            print(f"[{name}] WARNING: only {len(pool)} distinct training compounds -- proceeding "
                  f"anyway, but treat the hard-negative validation result with extra scrutiny.")
        print(f"[{name}] wrote eval_set.json to {result_dir}\n")
        coverage_report[name] = {"status": status, "n_training": len(pool), "n_training_lines": n_written}

    with open(os.path.join(OUT_DIR, "coverage_report.json"), "w") as f:
        json.dump(coverage_report, f, indent=2)
    print("=== COVERAGE REPORT ===")
    for name, r in coverage_report.items():
        print(name, r)
