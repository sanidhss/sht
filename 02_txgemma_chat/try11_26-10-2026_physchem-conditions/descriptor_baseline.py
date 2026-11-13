"""
Descriptor-only baseline for every fold -- no LLM, no GPU.

Purpose: establish a CEILING for what each necessary_condition can deliver. If a
mechanical property matcher cannot find the true drug from a condition, no text model
reading that same condition can either, and the fold's failure is a property of the
condition rather than of TxGemma.

Each filter is a hand translation of that fold's condition into RDKit descriptors,
written from the condition text alone and never from the true drug's structure.
"""
import json, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

LIB = "/work/hdd/bisa/sshah21/results/reasoning_screen/chembl_approved.json"
CHAINS = "/work/hdd/bisa/sshah21/results/reasoning_chains"

BASIC_N = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N[!#6]);!$(N=*);!$(NC=O);!$(Nc)]")
RING_N  = Chem.MolFromSmarts("[#7;R]")
ACID    = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
HALO    = Chem.MolFromSmarts("[F,Cl,Br,I]")
OH      = Chem.MolFromSmarts("[OX2H]")


def props(m):
    return dict(
        mw=Descriptors.MolWt(m), tpsa=rdMolDescriptors.CalcTPSA(m),
        logp=Descriptors.MolLogP(m), rotb=rdMolDescriptors.CalcNumRotatableBonds(m),
        arom=rdMolDescriptors.CalcNumAromaticRings(m),
        rings=rdMolDescriptors.CalcNumRings(m),
        basicN=m.HasSubstructMatch(BASIC_N), ringN=m.HasSubstructMatch(RING_N),
        acid=m.HasSubstructMatch(ACID), halo=m.HasSubstructMatch(HALO),
        nOH=len(m.GetSubstructMatches(OH)),
    )


# (keep?, sort key) per fold -- translated from each condition's own words.
FILTERS = {
    # "compact, highly polar, ring position able to carry positive charge"
    "fabry":   (lambda p: p["mw"] < 300 and p["ringN"] and p["arom"] == 0,
                lambda p: -p["tpsa"] / p["mw"]),
    # "small moderately lipophilic amphiphile, basic protonatable N, low TPSA, aromatic,
    #  not polyhydroxylated"
    "gaucher": (lambda p: p["mw"] < 400 and p["basicN"] and p["arom"] >= 1
                          and p["tpsa"] < 90 and p["nOH"] <= 2 and 1 <= p["logp"] <= 5,
                lambda p: -p["logp"]),
    # "rigid elongated hydrophobic, aromatic-rich, few rotatable bonds, low TPSA/MW,
    #  no basic amine"
    "cf":      (lambda p: p["arom"] >= 2 and p["rotb"] <= 8 and not p["basicN"]
                          and p["tpsa"] / p["mw"] < 0.22 and p["logp"] >= 3,
                lambda p: -p["logp"]),
    # "compact planar biaryl 250-400 Da, halogen/hydrophobic end, one acidic group,
    #  few rotatable bonds"
    "attr":    (lambda p: 250 <= p["mw"] <= 400 and p["arom"] >= 2 and p["acid"]
                          and p["rotb"] <= 6,
                lambda p: -p["logp"]),
}

lib = json.load(open(LIB))
print(f"{'fold':9s} {'true drug':14s} {'LLM rank':>12s} {'descriptor rank':>18s} {'pool kept':>10s}")
for fold, (keep, key) in FILTERS.items():
    chain = json.load(open(f"{CHAINS}/{fold}.json"))
    true = chain["reference_drug"].upper()
    rows = []
    for rec in lib + [{"name": chain["reference_drug"],
                       "smiles": dict(__import__("re").findall(r"(Compound \w+): (\S+)", chain["prompt"]))[
                           {v: k for k, v in chain["label_to_name"].items()}[chain["reference_drug"]]]}]:
        m = Chem.MolFromSmiles(rec.get("canonical_smiles") or rec["smiles"])
        if m is None:
            continue
        p = props(m)
        if keep(p):
            rows.append(((rec["name"] or "").upper(), key(p)))
    rows.sort(key=lambda r: r[1])
    rank = next((i for i, r in enumerate(rows, 1) if r[0] == true), None)
    print(f"{fold:9s} {chain['reference_drug']:14s} {'see try10':>12s} "
          f"{str(rank) + ' / ' + str(len(rows)) if rank else 'NOT KEPT':>18s} {len(rows):>10d}")
