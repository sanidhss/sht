"""
Output-resolution probe -- how many distinct scores can each trained adapter actually emit?

THIS IS A MEASUREMENT, NOT A FIX. It trains nothing and changes no training data. It
loads the four already-saved adapters and asks one question: across a few hundred
diverse molecules, how many different numbers does each model produce?

Why this question exists
------------------------
The hard-negative test uses 9 molecules per target. With 9 molecules, two very
different situations are indistinguishable:

  (a) Gaucher emitted 791 for six molecules because those six really are similar, or
  (b) Gaucher emitted 791 for six molecules because 791 is one of only a handful of
      values it can produce at all.

Under (b), "ambroxol 691 beats metformin 661" is a 30-point gap on an instrument whose
smallest real step might be ~100 points -- i.e. rounding noise presented as a ranking.
And it would matter well beyond this test: the downstream plan is to rank a large
library and forward a top-K shortlist to AF3 / PyRosetta / Boltz2. A model that emits
~15 distinct values across hundreds of compounds turns a 1,000-compound ranking into
~15 giant tie-groups, making any top-50 shortlist an arbitrary slice of a vast tie. No
amount of extra bioactivity data fixes that, because it is a property of how the model
emits scores, not of what it learned.

Observed numbers that motivated this (distinct predicted values across each target's
9 eval molecules):

    CF       7 of 9   ref 411 | easy 804,833,633,926,633 | hard 661,584,661
    Gaucher  4 of 9   ref 691 | easy 791,791,661,791,791 | hard 791,791,839
    Fabry    6 of 9   ref 436 | easy 697,860,744,742,685 | hard 436,436,436
    TTR      5 of 9   ref 781 | easy 918,720,994,660,994 | hard 781,781,781

Two distinct failures are visible there, and the existing pass/fail gate conflates them:
  Failure A -- within-family resolution: Fabry and TTR give the reference AND all three
    close analogs one identical value, while resolving unrelated drugs fine. They treat
    a whole structural family as a single object.
  Failure B -- output degeneracy: Gaucher dumps 791 onto six molecules and loses to
    metformin, at 1,208 training compounds. Not a data-volume problem, so extra data
    will not fix it.
This probe measures the capacity that underlies both.

Probe-set construction
----------------------
Pool: compounds present in CF/TTR/Fabry training data but NOT in Gaucher's (verified:
2,030 pooled distinct, only 30 overlap with Gaucher, so 2,000 are novel to it). Real,
measured, drug-like molecules already on disk -- no network, nothing downloaded.

Selection is MaxMin diversity picking on Morgan fingerprints, not random sampling.
MaxMin greedily takes the molecule furthest from everything already picked, so it spans
the pool. Random sampling could hand back a cluster by luck, and the entire result
hinges on the inputs being genuinely diverse.

Stratified, because a single aggregate number would not be interpretable:

  200  MaxMin-diverse, novel to Gaucher  -> resolution on unseen chemistry (the headline)
   50  random from the probed target's own training set
                                         -> MEMORIZATION CONTRAST. If a model resolves
                                            trained compounds finely but novel ones
                                            coarsely, that is recall, not resolution.
   50  highest-Tanimoto neighbours of the target's reference drug
                                         -> within-family resolution (Failure A) directly
    9  the existing eval molecules       -> anchors, so output ties back to known numbers

The probe set is built once with a fixed seed, saved to disk, and reused identically
for all four adapters -- otherwise cross-adapter comparison would be meaningless.

Each molecule is prompted against ITS OWN adapter's target protein sequence, recovered
from that target's eval_set.json by substituting the SMILES line of a known-good
prompt. Probing the Gaucher adapter with CFTR's sequence would measure nothing.

Metrics reported per adapter
----------------------------
  n_distinct / n_total           overall and per stratum
  value histogram                names the attractor values (is 791 one?)
  min_nonzero_gap                the instrument's real resolution -- THE number that
                                 decides whether 691-vs-661 is signal or rounding
  probe-set mean pairwise Tanimoto
                                 so if few values come out, "the inputs were all alike"
                                 can be ruled out as the cause
  n_unparseable                  generations that were not integers

KNOWN LIMITATION, stated so the number is not read as broader than it is: the pool is
bioactivity data for these four targets. It is drug-like and diverse within that space,
but it is NOT a random slice of chemical space and NOT a substitute for a real
DrugBank-wide screen. It is adequate for the question asked here -- can the model emit
many distinct values at all -- and not for estimating real screening performance.

A separate base-model load per adapter, rather than swapping adapters on one shared
base. try06's code records an unexplained backwards result (a memorized reference drug
scoring worse than unrelated negatives) from the shared-base approach that was never
root-caused. This probe's whole value is being trusted, so it pays ~35min/load x4
rather than reintroduce that doubt.
"""
import json
import os
import re
from collections import Counter

import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

RDLogger.DisableLog("rdApp.*")

MODEL = "google/txgemma-9b-predict"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
OUT_DIR = os.path.join(DATA_DIR, "resolution_probe")

TARGETS = ["gaucher", "cf", "fabry", "attr"]
N_DIVERSE = 200
N_HELD_IN = 50
N_NEIGHBOURS = 50
SEED = 0

# Distinct values each adapter produced across its 9 eval molecules, for the printed
# comparison. The probe asks whether these small counts reflect the model's ceiling.
EVAL_DISTINCT = {"cf": "7 of 9", "gaucher": "4 of 9", "fabry": "6 of 9", "attr": "5 of 9"}


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def compounds_of(target):
    """Distinct canonical SMILES in a target's augmented training file.

    The jsonl carries only {prompt, answer}, so compound identity is recovered from the
    prompt's SMILES line and canonicalized -- which also collapses the 5 augmented
    variants of each compound back together.
    """
    out = set()
    path = os.path.join(DATA_DIR, target, "train_blind.jsonl")
    for line in open(path):
        m = re.search(r"Drug SMILES: (.+)", json.loads(line)["prompt"])
        if not m:
            continue
        mol = Chem.MolFromSmiles(m.group(1).strip())
        if mol:
            out.add(Chem.MolToSmiles(mol))
    return out


def prompt_for(template_prompt, smiles):
    """Swap the SMILES into a known-good prompt, preserving that target's protein
    sequence and the exact instruction wording the model was trained on."""
    new, n = re.subn(r"(Drug SMILES: ).+", lambda m: m.group(1) + smiles,
                     template_prompt, count=1)
    if n != 1:
        raise ValueError("could not substitute SMILES into prompt template")
    return new


def build_probe_set():
    """Built once, saved, reused for every adapter -- cross-adapter comparison requires
    an identical molecule list."""
    per_target = {t: compounds_of(t) for t in TARGETS}
    for t, s in per_target.items():
        print(f"  {t}: {len(s)} distinct compounds in training file")

    # Pool novel to Gaucher: Gaucher is the target whose 791-clustering prompted this,
    # so the headline stratum must be unseen by it.
    pool = (per_target["cf"] | per_target["attr"] | per_target["fabry"]) - per_target["gaucher"]
    pool = sorted(pool)
    print(f"  pool novel to gaucher: {len(pool)}")

    mols = [Chem.MolFromSmiles(s) for s in pool]
    keep = [(s, m) for s, m in zip(pool, mols) if m is not None]
    pool, mols = [k[0] for k in keep], [k[1] for k in keep]
    fps = [fp(m) for m in mols]

    picker = MaxMinPicker()
    idx = list(picker.LazyBitVectorPick(fps, len(fps), N_DIVERSE, seed=SEED))
    diverse = [pool[i] for i in idx]
    print(f"  MaxMin-picked {len(diverse)} diverse novel compounds")

    return {"diverse_novel": diverse, "pool_size": len(pool)}


def mean_pairwise_tanimoto(smiles_list, cap=200):
    """Diversity of the probe set itself. If output values are few, this rules out
    'the inputs were all alike' as the explanation."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list[:cap]]
    fps = [fp(m) for m in mols if m is not None]
    sims, n = [], len(fps)
    for i in range(n):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return (sum(sims) / len(sims)) if sims else float("nan")


@torch.no_grad()
def score(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=8)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def to_int(s):
    try:
        return int(s.strip())
    except (ValueError, AttributeError, TypeError):
        return None


def analyse(name, values, label):
    """values: list of ints (unparseable already removed)."""
    if not values:
        return {"stratum": label, "n": 0}
    uniq = sorted(set(values))
    gaps = [b - a for a, b in zip(uniq, uniq[1:])]
    min_gap = min(gaps) if gaps else None
    counts = Counter(values)
    print(f"  [{name}/{label}] n={len(values)} distinct={len(uniq)} "
          f"({len(uniq)/len(values):.1%}) range={uniq[0]}-{uniq[-1]} "
          f"min_nonzero_gap={min_gap}")
    print(f"      top values: {counts.most_common(8)}")
    return {"stratum": label, "n": len(values), "n_distinct": len(uniq),
            "frac_distinct": len(uniq) / len(values), "min_nonzero_gap": min_gap,
            "value_min": uniq[0], "value_max": uniq[-1],
            "top_values": counts.most_common(15)}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Building probe set (once, reused for all adapters) ===")
    shared = build_probe_set()
    diverse = shared["diverse_novel"]
    div_tan = mean_pairwise_tanimoto(diverse)
    print(f"  mean pairwise Tanimoto of diverse stratum: {div_tan:.3f} "
          f"(lower = more diverse)")

    import random
    results = {}

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    for target in TARGETS:
        adapter_dir = os.path.join(DATA_DIR, target, "adapter", target)
        if not os.path.isdir(adapter_dir):
            print(f"[skip] {target}: no adapter at {adapter_dir}")
            continue

        eval_set = json.load(open(os.path.join(DATA_DIR, target, "eval_set.json")))
        template = eval_set["reference"]["prompt"]
        ref_smiles = eval_set["reference"]["smiles"]

        # Per-target strata. Held-in and neighbour strata are target-specific by
        # construction, so they are built here rather than in the shared set.
        own = sorted(compounds_of(target))
        held_in = random.Random(SEED).sample(own, min(N_HELD_IN, len(own)))

        ref_mol = Chem.MolFromSmiles(ref_smiles)
        pool_all = sorted(set(own) | set(diverse))
        pm = [(s, Chem.MolFromSmiles(s)) for s in pool_all]
        pm = [(s, m) for s, m in pm if m is not None]
        ref_fp = fp(ref_mol)
        sims = [(DataStructs.TanimotoSimilarity(ref_fp, fp(m)), s) for s, m in pm]
        sims.sort(reverse=True)
        neighbours = [s for _, s in sims[:N_NEIGHBOURS]]
        nb_sim_range = (round(sims[0][0], 3), round(sims[min(N_NEIGHBOURS, len(sims)) - 1][0], 3))

        anchors = ([eval_set["reference"]["smiles"]]
                   + [c["smiles"] for c in eval_set["easy_negatives"]]
                   + [c["smiles"] for c in eval_set["hard_negatives"]])

        strata = {"diverse_novel": diverse, "held_in": held_in,
                  "reference_neighbours": neighbours, "eval_anchors": anchors}
        total = sum(len(v) for v in strata.values())

        print(f"\n=== Loading base model + adapter: {target} "
              f"({total} molecules, eval-set distinct was {EVAL_DISTINCT.get(target,'?')}) ===",
              flush=True)
        print(f"  reference-neighbour Tanimoto range: {nb_sim_range[0]} down to {nb_sim_range[1]}")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="cuda:0", dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()

        per_stratum, raw, all_vals, unparseable = {}, {}, [], 0
        for label, smis in strata.items():
            vals = []
            raw[label] = {}
            for s in smis:
                g = score(model, tok, prompt_for(template, s))
                raw[label][s] = g
                v = to_int(g)
                if v is None:
                    unparseable += 1
                else:
                    vals.append(v)
            per_stratum[label] = analyse(target, vals, label)
            all_vals.extend(vals)

        overall = analyse(target, all_vals, "ALL")
        results[target] = {
            "n_molecules": total, "n_unparseable": unparseable,
            "eval_set_distinct": EVAL_DISTINCT.get(target),
            "diverse_mean_pairwise_tanimoto": div_tan,
            "reference_neighbour_tanimoto_range": nb_sim_range,
            "overall": overall, "per_stratum": per_stratum,
        }
        with open(os.path.join(OUT_DIR, f"{target}_raw_generations.json"), "w") as f:
            json.dump(raw, f, indent=2)

        del model, base
        torch.cuda.empty_cache()

        # Written after each adapter so a later OOM or timeout still leaves completed
        # adapters on disk.
        with open(os.path.join(OUT_DIR, "resolution_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== FINAL SUMMARY: output resolution per adapter ===")
    print(f"probe-set diversity (mean pairwise Tanimoto, diverse stratum): {div_tan:.3f}")
    for t, r in results.items():
        o = r["overall"]
        print(f"{t:8s} n={o['n']:4d} distinct={o['n_distinct']:4d} "
              f"({o['frac_distinct']:.1%}) min_gap={o['min_nonzero_gap']} "
              f"[eval-set was {r['eval_set_distinct']}]")
    print("\nHow to read this (full reasoning in this try's README):")
    print("  high distinct % + small min_gap -> fine resolution; the 9-molecule")
    print("     clustering was real chemistry, and rankings are meaningful")
    print("  low distinct % + large min_gap  -> a resolution CEILING. Small score")
    print("     differences (e.g. 691 vs 661) are rounding, library ranking collapses")
    print("     into few tie-groups, and more training data cannot fix it")
    print("  high on held_in but low on diverse_novel -> memorization, not resolution")
