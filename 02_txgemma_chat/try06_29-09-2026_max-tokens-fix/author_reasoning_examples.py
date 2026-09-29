"""
Authors the reasoning-chain training examples for the txgemma-chat leave-one-target-out
fine-tune. This is hand-written content, not generated or scraped -- each mechanism
description and reasoning fact is grounded in the real assay evidence gathered and
verified earlier (ChEMBL activity records, assay descriptions naming the actual
mechanism), the same standard of diligence used to verify ambroxol/lumacaftor/tafamidis
before trusting them as references.

Candidate pool per target = own real reference + own real hard negatives (from
eval_set.json) + the OTHER THREE targets' real reference drugs as cross-mechanism
distractors + two fixed easy negatives (aspirin, caffeine).

v2 (26-09-2026) -- redesigned after try04 showed the model was memorizing the single
fixed REASONING_CHAINS paragraph per target near-verbatim (confirmed by diffing its
generated text against the templates) rather than reasoning from the candidate SMILES.
Two changes:
  1. The completion is no longer one holistic paragraph about the top choice. It states
     an explicit NECESSARY CONDITION for the mechanism, then gives an explicit
     satisfies/fails/partial verdict + reason for EVERY candidate in the pool, checked
     against that condition -- the textual analogue of a link-prediction model checking
     a candidate embedding against a relation-specific compatibility criterion (what
     TxGNN does geometrically for indication/contraindication), but expressed so the
     model must apply a compositional rule (scaffold class x mechanism condition ->
     verdict) instead of reciting a memorized block. Because which candidates appear
     varies per augmented instance, the completion's actual content differs every time
     even for the same target -- verbatim block memorization stops being viable.
  2. Where try04 had exactly one hand-authored reasoning paragraph per target, the
     per-class WHY_CLASS_FAILS reasons are combinatorial (built from NECESSARY_CONDITION
     x SCAFFOLD_CLASS, ~12 pairs) rather than 4 monolithic strings, so there is no
     single fixed block long enough to memorize wholesale.

Output: one JSON file per target with the blinded prompt (sequence + mechanism
description + candidate list, disease/gene name never mentioned) and the authored
completion (necessary_condition + candidate_verdicts + ranking + self-contradiction
check). This is raw material for the leave-one-target-out training harness.
"""
import json
import os
import random
import zlib

BASE = "/projects/bisa/sshah21"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
OUT_DIR = "/work/hdd/bisa/sshah21/results/reasoning_chains"

EASY_NEGATIVES = {
    "aspirin":  "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}

REFERENCE_NAMES = {"fabry": "migalastat", "gaucher": "ambroxol", "cf": "lumacaftor", "attr": "tafamidis"}


def read_fasta(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


FASTA_PATHS = {
    "fabry": f"{BASE}/GLA.fasta", "gaucher": f"{BASE}/GBA1.fasta",
    "cf": f"{BASE}/CFTR.fasta", "attr": f"{BASE}/TTR.fasta",
}

# -----------------------------------------------------------------------------------
# Mechanism descriptions -- deliberately never name the gene, protein, or disease.
# Each is grounded in the real assay evidence already gathered for that reference drug.
# -----------------------------------------------------------------------------------
MECHANISM_DESCRIPTIONS = {
    "fabry": (
        "This is a lysosomal hydrolase. Destabilizing missense mutations in patients "
        "cause a lysosomal storage disorder. Its natural substrate is a glycosphingolipid; "
        "the enzyme specifically recognizes and cleaves the terminal alpha-linked sugar "
        "unit at the end of the substrate's sugar chain -- it does not interact with the "
        "lipid tail. A pharmacological chaperone for this enzyme is a small molecule that "
        "occupies the same catalytic pocket as this terminal sugar unit, adopting a "
        "similar ring shape, charge, and stereochemistry to the sugar's reaction "
        "transition state, binding reversibly to stabilize the misfolded mutant enzyme."
    ),
    "gaucher": (
        "This is a lysosomal hydrolase. Destabilizing missense mutations in patients "
        "cause a lysosomal storage disorder. Its natural substrate is a glycosphingolipid; "
        "the enzyme recognizes and cleaves the terminal beta-linked sugar unit at the end "
        "of the substrate's sugar chain. A pharmacological chaperone for this enzyme "
        "stabilizes the native fold of destabilized mutant forms in patient cells and "
        "restores residual catalytic activity; published cell-based assays on "
        "patient-derived cells describe this activity as modulation of the enzyme's "
        "activity rather than simple competitive occupation of the sugar-binding site, "
        "and report only modest potency."
    ),
    "cf": (
        "This is a large multi-pass membrane transport protein that must fold correctly "
        "and traffic to the cell surface after synthesis to function. The most common "
        "disease-causing mutation causes a folding defect: the protein is synthesized "
        "correctly at the sequence level but fails to fold properly and is degraded "
        "before ever reaching the cell membrane, so no functional protein is delivered "
        "to its working location. A corrector for this protein is a small molecule that "
        "binds the misfolded intermediate -- typically at a domain-domain interface "
        "within the protein, away from any catalytic or transport site -- and stabilizes "
        "a folding-competent conformation so more of the protein passes quality control "
        "and reaches the membrane."
    ),
    "attr": (
        "This protein normally circulates in blood plasma as a stable four-copy "
        "(tetrameric) assembly. Certain destabilizing mutations lower the energy barrier "
        "for the tetramer to dissociate into individual monomers; free monomers misfold "
        "and aggregate into pathological fibrils that deposit in tissues. A stabilizer "
        "for this protein binds within a pocket formed at the interface between two "
        "subunits of the native tetramer -- a pocket that normally binds a small "
        "hormone-like ligand as part of the protein's ordinary transport function -- "
        "kinetically locking the tetramer together and raising the barrier to "
        "dissociation, without altering any catalytic activity (this protein has none)."
    ),
}

# Paraphrases of the same mechanism -- same facts, different surface wording. Used only
# for training augmentation (never for the held-out eval example above), on the same
# principle as SMILES augmentation for the numeric task: don't let the model memorize
# one exact string, force it to generalize from the underlying content. Hand-written,
# not generated, so the facts stay exactly what was verified -- nothing new is claimed.
MECHANISM_PARAPHRASES = {
    "fabry": [
        "A lysosomal enzyme is deficient due to destabilizing missense mutations, "
        "causing a lysosomal storage disorder. The enzyme's natural substrate is a "
        "glycosphingolipid, and catalysis targets only the terminal alpha-linked sugar "
        "unit at the end of its sugar chain -- the lipid portion of the substrate is "
        "never engaged. Small-molecule chaperones for this enzyme work by binding in the "
        "same pocket normally occupied by that terminal sugar during the transition "
        "state of bond cleavage, matching its ring geometry, charge, and stereochemistry "
        "closely enough to reversibly stabilize the mutant enzyme's fold.",

        "Patients carry destabilizing missense mutations in a lysosomal hydrolase, "
        "resulting in a lysosomal storage disorder. This enzyme hydrolyzes a "
        "glycosphingolipid substrate by acting specifically on its terminal "
        "alpha-configured sugar residue; the ceramide/lipid segment of the substrate "
        "does not contact the enzyme. Chaperone therapy relies on small molecules that "
        "reversibly occupy the enzyme's catalytic site, structurally resembling the "
        "transition-state geometry, charge distribution, and stereochemical "
        "configuration of that terminal sugar, thereby stabilizing misfolded protein.",
    ],
    "gaucher": [
        "A lysosomal storage disorder results from destabilizing missense mutations in "
        "a lysosomal glycosidase. The enzyme's natural substrate is a glycosphingolipid, "
        "and it acts on the terminal beta-linked sugar residue of that substrate. "
        "Chaperone treatment stabilizes the native fold of destabilized mutant enzyme in "
        "patient cells, restoring residual activity; the published cell-based evidence "
        "characterizes this as a modulatory effect on enzyme activity rather than clean "
        "competitive occupation of the sugar-binding site, and reports only modest "
        "potency for the known chaperone.",

        "Destabilizing mutations in a lysosomal hydrolase cause a lysosomal storage "
        "disorder; the enzyme normally cleaves the terminal beta-configured sugar unit "
        "of a glycosphingolipid substrate. A pharmacological chaperone restores residual "
        "activity to mutant enzyme in patient-derived cells, though the documented "
        "mechanism for the known agent is described as modulatory rather than a clean "
        "sugar-mimicking active-site block, and its measured potency is modest rather "
        "than high.",
    ],
    "cf": [
        "A large multi-pass membrane transporter must fold correctly and traffic to the "
        "cell surface to function. Its most common disease mutation produces a folding "
        "defect -- the protein's sequence is correct, but it misfolds and is degraded "
        "before reaching the membrane, so no functional protein is ever delivered. "
        "Correctors are small molecules that bind this misfolded intermediate, typically "
        "at an interface between two of the protein's folded domains rather than at any "
        "catalytic or transport site, stabilizing a folding-competent shape so more "
        "protein passes quality control and reaches the cell surface.",

        "A multi-domain membrane transport protein requires correct folding and "
        "trafficking to reach the cell membrane. Its most common pathogenic mutation "
        "causes a folding defect: the sequence itself is unaffected, but the protein "
        "misfolds during biogenesis and is degraded, so none reaches its functional "
        "location. A small-molecule corrector binds the misfolded intermediate at a "
        "domain-domain interface, away from any catalytic or transport machinery, "
        "stabilizing a folding-competent conformation and increasing surface delivery.",
    ],
    "attr": [
        "A plasma protein normally exists as a stable tetramer (four identical "
        "subunits). Destabilizing mutations reduce the energy barrier to tetramer "
        "dissociation into monomers, which then misfold and aggregate into tissue-"
        "depositing fibrils. A stabilizing drug binds a pocket at the interface between "
        "two subunits of the native tetramer -- the same pocket that ordinarily binds a "
        "small hormone-like molecule as part of the protein's normal transport role -- "
        "kinetically locking the assembly together and raising the dissociation barrier, "
        "with no effect on any catalytic function (the protein has none).",

        "A blood plasma protein circulates as a four-subunit tetramer under normal "
        "conditions. Certain mutations destabilize this assembly, lowering the barrier "
        "to dissociation into monomeric subunits that misfold and form pathological "
        "fibrils in tissues. Stabilizer compounds occupy an interfacial pocket between "
        "two subunits -- normally the binding site for a small hormone-like ligand -- "
        "and kinetically stabilize the native tetramer against dissociation, without any "
        "effect on catalysis, since the protein carries out no catalytic reaction.",
    ],
}

# -----------------------------------------------------------------------------------
# The logical-elimination core (v2). Everything below is decomposed from the same
# facts try04's REASONING_CHAINS already established -- nothing new is being claimed,
# it's restructured into a checkable, compositional form: state the mechanism's
# necessary condition, classify each candidate's real scaffold, then check scaffold
# against condition. This is deliberately built the way a knowledge-graph model checks
# a candidate against a relation-specific compatibility criterion, so the transformer
# has to apply the same rule to each candidate rather than recite one fixed narrative.
# -----------------------------------------------------------------------------------
NECESSARY_CONDITION = {
    "fabry": "the candidate must mimic the shape, charge, and exact stereochemistry of "
             "a single small sugar ring at its reaction transition state -- compact, "
             "highly polar, with a ring position capable of carrying positive charge",
    "gaucher": "the candidate must act as a secondary/modulatory chaperone contact "
               "rather than a clean sugar-transition-state mimic -- the evidence here "
               "specifically describes modulation and modest potency, not direct "
               "high-potency sugar-site occupation",
    "cf": "the candidate must have size and shape complementarity to a "
          "protein-protein-like domain-domain interface -- a larger, rigid, "
          "multi-ring scaffold, not a compact polar transition-state-mimicking ring",
    "attr": "the candidate must mimic the shape, charge distribution, and size of a "
            "small hormone-like ligand at a hydrophobic interfacial pocket between two "
            "protein subunits, with a polar anchor at the entrance",
}

# What each reference compound's real scaffold actually looks like -- used both to
# describe "the reference" and to classify that same compound when it shows up as a
# cross-mechanism distractor in another target's candidate pool.
SCAFFOLD_CLASS = {
    "fabry": "a small six-membered ring with a basic ring nitrogen in place of the "
             "sugar's ring oxygen, decorated with hydroxyls matching the natural "
             "sugar's stereochemistry (an iminosugar)",
    "gaucher": "a simple aminocyclohexanol bearing a dihalogenated aniline group -- "
               "not itself a sugar-ring mimic",
    "cf": "a bis-aromatic molecule with a rigid spirocyclopropyl-linked bicyclic core",
    "attr": "a halogenated aromatic carboxylic-acid scaffold sized for a small "
            "hydrophobic channel with a polar anchor",
    "aspirin": "a small aromatic acetate ester (acetylsalicylic acid) with no ring "
               "nitrogen, no sugar-mimicking hydroxylation, and no rigid multi-ring "
               "interfacial scaffold",
    "caffeine": "a small planar xanthine (fused bicyclic purinedione) with no basic "
                "ring nitrogen positioned for transition-state charge mimicry and no "
                "interfacial anchoring group",
}

# Pairwise: why a compound belonging to scaffold class X fails target T's necessary
# condition. This is the explicit "contraindication" logic table -- the textual
# analogue of what a link-prediction model encodes geometrically as an incompatible
# drug-disease embedding. Every entry restates facts already established in
# NECESSARY_CONDITION/SCAFFOLD_CLASS above; nothing here is a new biochemical claim.
WHY_CLASS_FAILS = {
    "fabry": {
        "gaucher":  "an aminocyclohexanol-dihalogenated-aniline scaffold lacks the "
                    "sugar-ring hydroxylation pattern and transition-state charge "
                    "geometry this condition requires",
        "cf":       "a large rigid bis-aromatic bicyclic scaffold is sized for a "
                    "domain-domain interface, not the compact polar transition-state "
                    "pocket this condition requires",
        "attr":     "a halogenated aromatic carboxylic acid is sized for a hydrophobic "
                    "interfacial channel, not the compact polar sugar-mimicking pocket "
                    "this condition requires",
        "aspirin":  "a simple aromatic acetate ester has no ring nitrogen and no "
                    "sugar-like hydroxylation pattern to mimic the transition state",
        "caffeine": "a planar xanthine has no ring nitrogen positioned for charge "
                    "mimicry and no sugar-like hydroxylation",
    },
    "gaucher": {
        "fabry":    "a classic iminosugar is a textbook transition-state mimic, but "
                    "the evidence for this enzyme specifically describes modulatory "
                    "activity rather than clean sugar-site occupation, so a perfect "
                    "sugar mimic is a plausible-looking but evidentially unsupported "
                    "answer here",
        "cf":       "a large rigid bis-aromatic bicyclic scaffold does not resemble the "
                    "simple aminocyclohexanol contact chemistry the evidence describes",
        "attr":     "a halogenated aromatic carboxylic acid does not resemble the "
                    "simple aminocyclohexanol contact chemistry the evidence describes",
        "aspirin":  "a simple aromatic acetate ester has no aminocyclohexanol-type "
                    "contact chemistry",
        "caffeine": "a planar xanthine has no aminocyclohexanol-type contact chemistry",
    },
    "cf": {
        "fabry":    "a small polar iminosugar is built to fill a compact "
                    "transition-state pocket, irrelevant to stabilizing a large "
                    "multi-domain interface with no sugar-binding function",
        "gaucher":  "a small aminocyclohexanol is sized for a compact secondary "
                    "contact site, not a large rigid domain-domain interface",
        "attr":     "a halogenated aromatic carboxylic acid is sized for a "
                    "hormone-binding channel between two folded subunits, not the "
                    "interdomain interface of a single misfolded protein this "
                    "condition requires",
        "aspirin":  "a small planar acetate ester is far too small and flexible to "
                    "wedge into and rigidify a domain-domain interface",
        "caffeine": "a small planar xanthine is far too small and flexible to wedge "
                    "into and rigidify a domain-domain interface",
    },
    "attr": {
        "fabry":    "a small polar iminosugar is shaped and sized for a "
                    "monosaccharide-binding glycosidase pocket, not the hydrophobic "
                    "hormone-binding channel between two folded subunits this "
                    "condition requires",
        "gaucher":  "a small aminocyclohexanol lacks the halogenated-aromatic "
                    "hydrophobic bulk needed to occupy a hormone-binding channel",
        "cf":       "a bis-aromatic spirocyclopropyl scaffold is built for a "
                    "domain-domain interface within a single misfolded protein, not "
                    "the inter-subunit hormone-binding channel this condition requires",
        "aspirin":  "a small aromatic acetate ester lacks the halogenation and "
                    "hydrophobic bulk to occupy the hormone-binding channel",
        "caffeine": "a small planar xanthine lacks the halogenation and hydrophobic "
                    "bulk to occupy the hormone-binding channel",
    },
}

# Own hard negatives are real, structurally near-identical analogs of the reference
# (same broad scaffold class) with a genuinely different measured potency -- a
# deliberately harder case than a wrong scaffold class entirely: they PARTIALLY satisfy
# the condition (coarse scaffold match) but the mechanism's dependence on exact
# stereochemistry/substitution is what would need independent confirmation. This is
# templated from SCAFFOLD_CLASS, not a per-compound claim about which exact atom
# differs (that isn't something to assert without the specific structural diff verified).
def own_hard_negative_reason(target):
    return (f"same broad scaffold class as the reference -- {SCAFFOLD_CLASS[target]} -- "
            f"so it coarsely satisfies the condition; it is a real, distinct "
            f"stereoisomer/analog, and the mechanism's dependence on exact "
            f"stereochemistry or substitution pattern (not just the general scaffold) "
            f"is what would need independent confirmation before ranking it above the "
            f"reference")


SELF_CHECK_TEMPLATES = [
    "This would be wrong if the necessary condition above were not actually "
    "\"{cond}\", or if the top choice's scaffold turned out not to satisfy it after "
    "all -- neither applies given the stated mechanism.",
    "This reasoning fails if the stated mechanism does not really require that {cond}, "
    "or if the top-ranked candidate's scaffold class were misidentified above -- "
    "neither is the case here.",
]


def classify_candidate(target, name):
    """Returns (scaffold_key, satisfies) for a candidate `name` in target `target`'s
    pool. scaffold_key indexes into SCAFFOLD_CLASS / WHY_CLASS_FAILS (except for the
    target's own hard negatives, handled separately). satisfies is True / False /
    "partial"."""
    if name == REFERENCE_NAMES[target]:
        return target, True
    if name.startswith(f"{target}_analog"):
        return "own_hard_negative", "partial"
    for other, ref in REFERENCE_NAMES.items():
        if other != target and name == ref:
            return other, False
    if name in EASY_NEGATIVES:
        return name, False
    raise ValueError(f"unrecognized candidate {name!r} for target {target!r}")


def build_candidate_verdicts(target, names_list, name_to_label):
    verdicts = {}
    for name in names_list:
        scaffold_key, satisfies = classify_candidate(target, name)
        label = name_to_label[name]
        if scaffold_key == "own_hard_negative":
            reason = own_hard_negative_reason(target)
        elif satisfies is True:
            reason = (f"matches {SCAFFOLD_CLASS[target]}, exactly the profile the "
                      f"necessary condition calls for")
        else:
            reason = WHY_CLASS_FAILS[target][scaffold_key]
        verdicts[label] = {"satisfies_condition": satisfies, "reason": reason}
    return verdicts


def build_completion(target, names_list, name_to_label, rng):
    ref_label = name_to_label[REFERENCE_NAMES[target]]
    ranking = [ref_label] + [name_to_label[n] for n in names_list if n != REFERENCE_NAMES[target]]
    verdicts = build_candidate_verdicts(target, names_list, name_to_label)
    self_check = rng.choice(SELF_CHECK_TEMPLATES).format(cond=NECESSARY_CONDITION[target])
    return {
        "necessary_condition": NECESSARY_CONDITION[target],
        "candidate_verdicts": verdicts,
        "ranking": ranking,
        "top_choice": ref_label,
        "self_contradiction_check": self_check,
    }


def blinded_labels(seed_key, names):
    rng = random.Random(zlib.crc32(seed_key.encode()))
    shuffled = names[:]
    rng.shuffle(shuffled)
    labels = [f"Compound {chr(ord('A') + i)}" for i in range(len(shuffled))]
    return dict(zip(labels, shuffled))


def build_prompt(seq, mechanism, label_to_name, candidates):
    compound_lines = "\n".join(f"{label}: {candidates[name]}" for label, name in label_to_name.items())
    return f"""You are analyzing candidate small molecules as pharmacological chaperones/correctors/stabilizers for a human protein.

Protein sequence (single-letter amino acids):
{seq}

Biochemical context:
{mechanism}

Candidate compounds (SMILES):
{compound_lines}

Task: first state the ONE necessary structural/physicochemical condition a candidate must satisfy given this mechanism (not any memorized drug or brand name association). Then, for EVERY candidate listed above, check its real structure against that condition and give an explicit verdict: does it satisfy the condition, fail it, or only partially satisfy it (e.g. same broad scaffold class but a specific point unconfirmed) -- with a one-line structural reason for each. Only after checking every candidate individually, rank them from most to least likely to be the real pharmacological agent, and state what evidence would contradict your top choice.

Respond with strict JSON only, no other text, in this exact schema:
{{"necessary_condition": "...", "candidate_verdicts": {{"<label>": {{"satisfies_condition": true|false|"partial", "reason": "..."}}, "...": {{...}}}}, "ranking": ["<label>", "..."], "top_choice": "<label>", "self_contradiction_check": "..."}}
"""


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # real SMILES for each target's own reference + hard negatives, from eval_set.json
    eval_sets = {}
    for name in REFERENCE_NAMES:
        eval_sets[name] = json.load(open(os.path.join(DATA_DIR, name, "eval_set.json")))

    reference_smiles = {name: es["reference"]["smiles"] for name, es in eval_sets.items()}

    for name in REFERENCE_NAMES:
        es = eval_sets[name]
        seq = read_fasta(FASTA_PATHS[name])

        candidates = {REFERENCE_NAMES[name]: es["reference"]["smiles"]}
        for i, hn in enumerate(es["hard_negatives"]):
            candidates[f"{name}_analog_{i+1}"] = hn["smiles"]
        for other, smi in reference_smiles.items():
            if other != name:
                candidates[REFERENCE_NAMES[other]] = smi
        candidates.update(EASY_NEGATIVES)

        names = list(candidates.keys())
        label_to_name = blinded_labels(name, names)
        name_to_label = {v: k for k, v in label_to_name.items()}

        prompt = build_prompt(seq, MECHANISM_DESCRIPTIONS[name], label_to_name, candidates)
        rng = random.Random(zlib.crc32(f"{name}_held_out_eval".encode()))
        completion = build_completion(name, names, name_to_label, rng)

        out = {
            "target": name,
            "reference_drug": REFERENCE_NAMES[name],
            "label_to_name": label_to_name,
            "prompt": prompt,
            "completion": completion,
        }
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{name}] wrote reasoning example: {len(names)} candidates "
              f"(1 reference + {len(es['hard_negatives'])} own hard negatives + "
              f"{len(reference_smiles) - 1} cross-target references + {len(EASY_NEGATIVES)} easy negatives), "
              f"top_choice label = {name_to_label[REFERENCE_NAMES[name]]}")
