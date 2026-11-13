"""
Authors the reasoning-chain training examples for the txgemma-chat leave-one-target-out
fine-tune. This is hand-written content, not generated or scraped -- each mechanism
description and reasoning chain is grounded in the real assay evidence gathered and
verified earlier (ChEMBL activity records, assay descriptions naming the actual
mechanism), the same standard of diligence used to verify ambroxol/lumacaftor/tafamidis
before trusting them as references.

Candidate pool per target = own real reference + own real hard negatives (from
eval_set.json) + the OTHER THREE targets' real reference drugs as cross-mechanism
distractors + two fixed easy negatives (aspirin, caffeine). The cross-mechanism
distractors are the important addition: they test whether the model can recognize that
a real, potent chaperone for a DIFFERENT mechanism is wrong here -- a much sharper test
than unrelated drugs, and it directly probes the thing we actually care about
(mechanism-specific reasoning, not "does this look drug-like").

Output: one JSON file per target with the blinded prompt (sequence + mechanism
description + candidate list, disease/gene name never mentioned) and the authored
completion (ranking + reasoning chain + self-contradiction check). This is raw
material for the leave-one-target-out training harness (not built yet).
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
# Reasoning chains -- authored, grounded in the mechanism above and the real structural
# facts of each reference compound. Each explicitly cites the STATED mechanism (not an
# assumed disease identity) and includes a self-contradiction check.
# -----------------------------------------------------------------------------------
REASONING_CHAINS = {
    "fabry": (
        "The stated mechanism requires mimicking the shape, charge, and exact "
        "stereochemistry of a single small sugar ring at its reaction transition state "
        "-- not the size or complexity of the whole natural substrate, whose lipid "
        "portion the enzyme never contacts. The reference compound is a small "
        "six-membered ring with a basic nitrogen in place of the sugar's ring oxygen "
        "(protonated at lysosomal pH, mimicking the transition state's developing "
        "positive charge) decorated with hydroxyls matching the natural sugar's "
        "stereochemistry -- exactly the profile the stated mechanism calls for. Larger, "
        "more aromatic candidates rely on ring-stacking and hydrophobic pocket-filling, "
        "a strategy suited to open hydrophobic pockets (e.g. kinase ATP sites), not the "
        "compact, highly polar transition-state pocket described here. "
        "Self-contradiction check: this reasoning would be wrong if the binding pocket "
        "were shown to be large and hydrophobic rather than compact and polar, or if the "
        "mechanism did not depend on transition-state charge mimicry -- neither applies "
        "to the mechanism as stated."
    ),
    "gaucher": (
        "The stated mechanism describes the same broad enzyme class as a terminal-sugar "
        "glycosidase, but explicitly notes the real chaperone's activity is reported as "
        "modulation rather than simple sugar-mimicry occupation, and that its potency is "
        "only modest -- this is a harder case than one where the correct answer is a "
        "textbook small iminosugar. The reference compound is not itself a sugar-ring "
        "mimic: it is a simple aminocyclohexanol with a dihalogenated aniline group, "
        "consistent with a distinct, secondary chaperone contact rather than direct "
        "transition-state occupation, which matches the stated evidence more closely "
        "than a candidate built to be a perfect sugar mimic would. Candidates built "
        "specifically to mimic a DIFFERENT enzyme's sugar transition state (small "
        "polyhydroxylated rings with a ring nitrogen at a different position) are "
        "plausible-looking but are shaped for a different, specific sugar stereochemistry "
        "and pocket, and the evidence here does not describe that same clean mimicry "
        "mechanism. Self-contradiction check: this would be wrong if stronger evidence "
        "showed direct competitive occupation of the sugar site with high potency -- the "
        "stated evidence instead describes modest, modulatory activity, which is the "
        "basis for preferring this candidate over a cleaner but unsupported sugar-mimic "
        "alternative."
    ),
    "cf": (
        "This mechanism has no catalytic pocket or transition state to imitate at all -- "
        "the defect is a kinetic folding/trafficking problem, and the corrector's job is "
        "to stabilize a domain-domain interface of the full-length misfolded protein "
        "during biogenesis. Neither 'small and polar' nor 'mimics a small metabolite' is "
        "relevant here; what matters is size and shape complementarity to a specific "
        "protein-protein-like interface, which typically requires a larger, more rigid, "
        "multi-ring scaffold than a transition-state mimic. The reference compound is a "
        "bis-aromatic molecule with a rigid spirocyclopropyl-linked bicyclic core, "
        "consistent with a scaffold built to wedge into and rigidify a specific "
        "domain interface -- not merely 'complex-looking' in a generic sense, but sized "
        "for this specific structural role. Small polyhydroxylated sugar-mimic-type "
        "candidates are poorly matched here: their small size and polarity, an advantage "
        "for filling a compact sugar-binding pocket, is irrelevant to stabilizing a large "
        "multi-domain interface with no sugar-binding function at all. "
        "Self-contradiction check: this would be wrong if the defect described were "
        "instead a catalytic or substrate-turnover problem rather than a folding defect "
        "-- but the mechanism explicitly describes a folding/trafficking defect, not "
        "substrate turnover."
    ),
    "attr": (
        "This mechanism requires binding an existing interfacial pocket that normally "
        "accommodates a specific small hormone-like ligand, at the junction between two "
        "protein subunits -- not a catalytic site, since the protein has no catalytic "
        "activity to begin with. The reference compound is a halogenated aromatic "
        "carboxylic-acid scaffold sized for a small hydrophobic hormone-binding channel "
        "with a polar anchor at the entrance, consistent with occupying this specific "
        "interfacial site and locking the two subunits together. Small polyhydroxylated "
        "sugar-mimic candidates are shaped and sized for an entirely different job -- a "
        "monosaccharide-binding glycosidase pocket -- not a hydrophobic hormone channel "
        "between two folded protein subunits, and there is no catalytic transition state "
        "here to mimic in the first place. Self-contradiction check: this would be wrong "
        "if the described defect were instead a catalytic or trafficking problem rather "
        "than oligomer dissociation -- but the mechanism explicitly describes a stable "
        "tetramer dissociating into monomers, which calls for a stabilizer, not a "
        "chaperone or corrector."
    ),
}


def split_reasoning(chain_text):
    """Splits an authored REASONING_CHAINS entry into (reasoning, self_check) to match
    the 4-field schema the prompt actually asks for -- every chain ends with a
    'Self-contradiction check: ...' sentence."""
    marker = "Self-contradiction check:"
    idx = chain_text.index(marker)
    return chain_text[:idx].strip(), chain_text[idx + len(marker):].strip()


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

Task: using ONLY the biochemical reasoning above (mechanism, pocket type, shape/charge complementarity) -- not any memorized drug or brand name association -- rank the candidates from most to least likely to be the real pharmacological agent for this specific protein, and explain why. Explicitly state what evidence would contradict your top choice.

Respond with strict JSON only, no other text, in this exact schema:
{{"ranking": ["<label>", "..."], "top_choice": "<label>", "reasoning": "...", "self_contradiction_check": "..."}}
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
        reasoning, self_check = split_reasoning(REASONING_CHAINS[name])
        completion = {
            "ranking": [name_to_label[REFERENCE_NAMES[name]]] + [
                name_to_label[n] for n in names if n != REFERENCE_NAMES[name]
            ],
            "top_choice": name_to_label[REFERENCE_NAMES[name]],
            "reasoning": reasoning,
            "self_contradiction_check": self_check,
        }

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
