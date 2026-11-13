"""
Zero-shot base-model scoring of all four reference drugs -- the decisive test for the
one remaining hypothesis about why TTR fails.

WHY THIS EXISTS
---------------
try11 produced a clean positive for Gaucher (3/3 seeds pass leak-free; ambroxol ranks
~1st of 500 decoys, scoring 478 vs a decoy median of 691) and no signal at all for TTR
(tafamidis 815.7 vs a decoy median of 815; 254 of 500 decoys beat it).

Four structural explanations for that gap were tested against the data and all four are
dead, because in every case Gaucher succeeds with LESS of the thing the explanation says
TTR lacks:

  1. "The model can't emit values as low as 552."   TTR's own labels span 0-1000 and
     33.7% of them are below 552.
  2. "Tafamidis's chemical class is missing from training."   It is -- 0 of 476 TTR
     compounds share its Bemis-Murcko scaffold. But 0 of 1,202 Gaucher compounds share
     ambroxol's either, and Gaucher works.
  3. "The scaffold holdout stripped tafamidis's nearest neighbours."   It did, but
     ambroxol has strictly less structural support (best Tanimoto 0.173 vs tafamidis's
     0.340) and still wins.
  4. "Not enough data (476 vs 1,202 compounds)."   try07 already downsampled Gaucher to
     493 compounds and it still passed 2/3 seeds.

What is left is the one explanation that does NOT live in the fine-tuning data at all:
PRIOR FAMILIARITY IN THE BASE MODEL. Ambroxol is a ubiquitous over-the-counter mucolytic
with decades of literature; tafamidis is a 2011 orphan drug. If TxGemma's pretraining
already encodes ambroxol's activity profile, then LoRA only has to not destroy that,
whereas for tafamidis it would have to construct the answer from 476 compounds whose
nearest relative is a 0.34-Tanimoto stranger. That would explain the one thing no
structural account can: why the drug with LESS support is the one that works.

Crucially, this hypothesis predicts an effect that is measurable WITHOUT ANY TRAINING.

WHAT THIS RUN MEASURES
----------------------
One base-model load, no LoRA, no gradients, no epochs, nothing learned -- so none of the
three bugs this project has fought (training leak, over-training, greedy-decoding
artifact) can apply, and there is no checkpoint to select.

Two independent readouts per molecule, on exactly try11's molecule sets:

  A. AFFINITY, zero-shot. Score the reference drug, its 3 scaffold-held hard negatives,
     and the same 500 MaxMin-diverse cross-target decoys try11 used (same builder, same
     seed=0, so the decoy sets are identical and the percentiles are directly
     comparable). Reported as an enrichment percentile, the same gate try11 used.

     Prediction if familiarity is the cause: ambroxol already separates from the decoy
     field BEFORE fine-tuning, and tafamidis does not.

  B. SMILES FAMILIARITY, readout-independent. Per-token negative log-likelihood of each
     molecule's own SMILES string under the base model. This asks "how predictable is
     this molecule's structure to a model that has read the literature," and it does not
     touch the affinity head, the prompt template, or the digit tokens at all. A molecule
     seen often in pretraining is cheaper to spell.

     B exists because A alone cannot distinguish "the base model knows ambroxol" from
     "the base model knows neither drug." If zero-shot affinity is flat for both, B is
     the only thing left that can still separate them -- and it is one forward pass per
     molecule, cheaper than the generate call it rides along with.

     Each drug's NLL is reported as a PERCENTILE against its own 500-decoy field, not as
     a raw number, because per-token NLL is sensitive to SMILES length and composition
     and raw values across different molecules are not comparable.

HOW TO READ THE RESULT
----------------------
  - Ambroxol strong on A and/or B, tafamidis weak on both  -> prior familiarity is the
    cause. TTR's failure is not fixable with more TTR data; it needs either a target
    where the base model already has a prior, or a different mechanism for injecting one.
  - Both drugs flat on A and B  -> familiarity is NOT the cause either, and the real
    difference has to be in what fine-tuning did, not in what the model arrived knowing.
    That is a different and narrower search than the one this run closes.
  - Tafamidis strong zero-shot and weak after fine-tuning -> fine-tuning DESTROYED a
    working prior, which would make TTR a training-damage problem, not a knowledge gap.

Any of those three is a real answer. The point of this run is that it cannot come back
uninformative.

SCOPE
-----
All four targets in ONE job, deliberately. The try05 scope-coupling bug was about a
script silently training on whichever targets a shared report marked ready; here nothing
is trained and nothing is written to any per-target directory, so the targets are read
only and the sequential loop is safe. TARGETS can still be set explicitly to narrow it.
One base-model load is reused across all four, which is why this costs ~2h rather than
try08's 1h34m for four loads and a third of the molecules.
"""
import json
import os
import re
from collections import defaultdict

import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from transformers import AutoTokenizer, AutoModelForCausalLM

RDLogger.DisableLog("rdApp.*")

MODEL = "google/txgemma-9b-predict"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
OUT_DIR = os.path.join(DATA_DIR, "zero_shot_familiarity")

ALL_TARGETS = ["gaucher", "cf", "fabry", "attr"]
TARGETS = [t for t in os.environ.get("TARGETS", ",".join(ALL_TARGETS)).split(",") if t]
N_DECOYS = int(os.environ.get("N_DECOYS", "500"))


# ---------------------------------------------------------------- shared with try11
# Copied verbatim rather than imported: try folders snapshot the exact code that ran, and
# an import would let a later edit to try11 silently change what this run means.
def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def scaffold_of(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


_GROUP_CACHE = {}


def load_grouped(target):
    """Memoized because this run touches all four targets in one process and the decoy
    builder reads every OTHER target's file -- 20 parses of up to 17MB otherwise, all
    identical. The function is read-only and deterministic, so caching cannot change any
    result; it only avoids re-canonicalizing the same ~7,000 SMILES twenty times."""
    if target in _GROUP_CACHE:
        return _GROUP_CACHE[target]
    groups = defaultdict(list)
    for line in open(os.path.join(DATA_DIR, target, "train_blind.jsonl")):
        d = json.loads(line)
        m = re.search(r"Drug SMILES: (.+)", d["prompt"])
        if not m:
            continue
        c = canon(m.group(1).strip())
        if c:
            groups[c].append(d)
    _GROUP_CACHE[target] = groups
    return groups


def prompt_for(template, smiles):
    new, n = re.subn(r"(Drug SMILES: ).+", lambda m: m.group(1) + smiles, template, count=1)
    if n != 1:
        raise ValueError("SMILES substitution failed")
    return new


def digit_token_map(tok):
    m = {}
    for d in range(10):
        ids = tok(str(d), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            return None
        m[d] = ids[0]
    return m


@torch.no_grad()
def score_both(model, tok, prompt, dmap):
    """Greedy AND digit-logit expectation from ONE generate call -- try11's readout,
    unchanged, so zero-shot and fine-tuned numbers differ only by the adapter."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    plen = ids["input_ids"].shape[1]
    text = tok.decode(out.sequences[0][plen:], skip_special_tokens=True).strip()
    try:
        greedy = int(text)
    except (ValueError, TypeError):
        greedy = None

    digit_exp = None
    if dmap is not None and greedy is not None and out.scores:
        width = max(1, min(4, len(str(greedy))))
        if len(out.scores) >= width:
            did = torch.tensor([dmap[d] for d in range(10)], device=out.scores[0].device)
            total, place = 0.0, 10 ** (width - 1)
            for pos in range(width):
                probs = torch.softmax(out.scores[pos][0][did].float(), dim=-1)
                total += float(sum(p * d for d, p in enumerate(probs.tolist()))) * place
                place //= 10
            digit_exp = total
    return greedy, digit_exp


def build_decoys(target, n):
    own = set(load_grouped(target).keys())
    pool = set()
    for t in ALL_TARGETS:
        if t != target:
            pool |= set(load_grouped(t).keys())
    pool = sorted(pool - own)
    mols = [(s, Chem.MolFromSmiles(s)) for s in pool]
    mols = [(s, m) for s, m in mols if m is not None]
    fps = [fp(m) for _, m in mols]
    k = min(n, len(fps))
    idx = list(MaxMinPicker().LazyBitVectorPick(fps, len(fps), k, seed=0))
    return [mols[i][0] for i in idx], len(pool)


def enrichment(ref, vals):
    vv = [v for v in vals if v is not None]
    if ref is None or not vv:
        return None
    better = sum(1 for v in vv if v < ref)
    return {"rank": better, "n": len(vv), "percentile": 100.0 * better / len(vv),
            "reference_score": ref}


# ---------------------------------------------------------------- new in this try
@torch.no_grad()
def smiles_nll(model, tok, smiles):
    """Per-token negative log-likelihood of the raw SMILES string itself.

    No prompt, no protein, no digits -- just "how surprised is the base model by this
    molecule's spelling." Length-normalized so longer SMILES are not automatically
    scored as less familiar, though it is still only meaningful as a percentile against
    a decoy field of comparable molecules, which is how it is reported.
    """
    ids = tok(smiles, return_tensors="pt").to(model.device)
    if ids["input_ids"].shape[1] < 2:
        return None
    out = model(**ids, labels=ids["input_ids"])
    return float(out.loss.item())


def pct_below(ref, vals):
    """Percentile of ref within vals, counting how many decoys are LOWER. For NLL, lower
    = more familiar, so a LOW percentile means the drug is unusually familiar."""
    vv = [v for v in vals if v is not None]
    if ref is None or not vv:
        return None
    below = sum(1 for v in vv if v < ref)
    return {"rank": below, "n": len(vv), "percentile": 100.0 * below / len(vv),
            "reference_value": ref}


def med(vals):
    vv = sorted(v for v in vals if v is not None)
    return vv[len(vv) // 2] if vv else None


if __name__ == "__main__":
    bad = [t for t in TARGETS if t not in ALL_TARGETS]
    if bad:
        raise SystemExit(f"unknown target(s) {bad}; valid: {ALL_TARGETS}")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== zero-shot base-model scoring: {TARGETS} ===", flush=True)
    print("no LoRA, no training, no gradients -- the untouched base model\n", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dmap = digit_token_map(tok)
    print(f"per-digit tokenization: {'yes' if dmap else 'NO -- digit expectation disabled'}")

    # ONE load, reused for every target. Nothing is modified, so there is no adapter
    # state to leak between targets -- the reason try08 reloaded per adapter does not
    # apply to a model that is never written to.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="cuda:0", dtype=torch.bfloat16)
    model.eval()
    print("base model loaded\n", flush=True)

    results = {}
    for target in TARGETS:
        eval_set = json.load(open(os.path.join(DATA_DIR, target, "eval_set.json")))
        template = eval_set["reference"]["prompt"]
        ref_smiles = canon(eval_set["reference"]["smiles"])
        ref_name = eval_set["reference"].get("name") or target
        ref_real = eval_set["reference"]["score"]
        hard = [{"smiles": canon(c["smiles"]), "real": c["score"],
                 "name": c.get("name") or c["smiles"][:20]}
                for c in eval_set["hard_negatives"]]

        groups = load_grouped(target)
        decoys, pool_size = build_decoys(target, N_DECOYS)
        # Reported for the record: zero-shot has no training set, so "held out" is
        # meaningless here -- every molecule is equally unseen by this model. Printed so a
        # later reader does not have to wonder whether a holdout was applied.
        nn_support = None
        rm = Chem.MolFromSmiles(ref_smiles)
        if rm is not None and groups:
            rfp = fp(rm)
            sims = []
            for c in groups:
                cm = Chem.MolFromSmiles(c)
                if cm is not None:
                    sims.append(DataStructs.TanimotoSimilarity(rfp, fp(cm)))
            nn_support = max(sims) if sims else None

        print(f"\n{'=' * 70}")
        print(f"=== {target}: {ref_name} (real {ref_real}) ===")
        print(f"{len(groups)} compounds exist in this target's training file but NONE are "
              f"used -- nothing is trained here.")
        print(f"best Tanimoto from {ref_name} to that unused pool: "
              f"{nn_support if nn_support is None else round(nn_support, 3)}")
        print(f"decoys: {len(decoys)} MaxMin-diverse from {pool_size} cross-target "
              f"compounds (same builder and seed as try11)", flush=True)

        # --- A. affinity, zero-shot
        ref_g, ref_d = score_both(model, tok, prompt_for(template, ref_smiles), dmap)
        print(f"  [A] {ref_name}: greedy={ref_g} "
              f"digit_exp={round(ref_d, 1) if ref_d is not None else None} "
              f"(real {ref_real})", flush=True)

        hard_scored = []
        for h in hard:
            g, d = score_both(model, tok, prompt_for(template, h["smiles"]), dmap)
            hard_scored.append({"name": h["name"], "real": h["real"],
                                "greedy": g, "digit_exp": d})
            print(f"  [A] hard neg {h['name']} (real {h['real']}): greedy={g} "
                  f"digit_exp={round(d, 1) if d is not None else None}", flush=True)

        dg, dd = [], []
        for i, s in enumerate(decoys):
            g, d = score_both(model, tok, prompt_for(template, s), dmap)
            dg.append(g)
            dd.append(d)
            if (i + 1) % 100 == 0:
                print(f"    [A] {i + 1}/{len(decoys)} decoys", flush=True)

        # --- B. SMILES familiarity
        ref_nll = smiles_nll(model, tok, ref_smiles)
        hard_nll = [smiles_nll(model, tok, h["smiles"]) for h in hard]
        decoy_nll = []
        for i, s in enumerate(decoys):
            decoy_nll.append(smiles_nll(model, tok, s))
            if (i + 1) % 250 == 0:
                print(f"    [B] {i + 1}/{len(decoys)} decoy NLLs", flush=True)

        res = {"target": target, "reference_name": ref_name, "reference_real": ref_real,
               "reference_smiles": ref_smiles, "n_decoys": len(decoys),
               "decoy_pool_size": pool_size,
               "n_compounds_in_unused_training_file": len(groups),
               "best_tanimoto_to_unused_pool": nn_support,
               "hard_detail": hard_scored,
               "smiles_nll": {
                   "reference": ref_nll,
                   "hard": [{"name": h["name"], "nll": n}
                            for h, n in zip(hard, hard_nll)],
                   "decoy_median": med(decoy_nll),
                   "percentile": pct_below(ref_nll, decoy_nll),
               }}
        for name, refv, dvals in (("greedy", ref_g, dg), ("digit_exp", ref_d, dd)):
            beats = (refv is not None and
                     all(h[name] is None or refv < h[name] for h in hard_scored))
            collapsed = (refv is not None and all(h[name] == refv for h in hard_scored))
            e = enrichment(refv, dvals)
            res[name] = {"reference": refv, "beats_hard": beats, "collapsed": collapsed,
                         "enrichment": e, "decoy_median": med(dvals),
                         "n_distinct_decoys": len({round(v) for v in dvals
                                                   if v is not None}),
                         "hard": [h[name] for h in hard_scored]}
            if e:
                print(f"  [A] ENRICHMENT [{name}]: {e['rank']}/{e['n']} = "
                      f"{e['percentile']:.1f} percentile | ref {refv if refv is None else round(refv, 1)} "
                      f"vs decoy median {res[name]['decoy_median']} | "
                      f"beats_hard={beats} | distinct_decoys="
                      f"{res[name]['n_distinct_decoys']}", flush=True)
        p = res["smiles_nll"]["percentile"]
        if p:
            print(f"  [B] SMILES NLL: {ref_nll:.4f} vs decoy median "
                  f"{res['smiles_nll']['decoy_median']:.4f} -> "
                  f"{p['percentile']:.1f} percentile "
                  f"(LOW = unusually familiar to the base model)", flush=True)

        results[target] = res
        with open(os.path.join(OUT_DIR, f"{target}_zero_shot_scores.json"), "w") as f:
            json.dump({"decoys": decoys, "greedy": dg, "digit_exp": dd,
                       "smiles_nll": decoy_nll}, f)
        with open(os.path.join(OUT_DIR, "zero_shot_summary.json"), "w") as f:
            json.dump({"targets": TARGETS, "n_decoys": N_DECOYS, "model": MODEL,
                       "results": results}, f, indent=2)

    print(f"\n\n{'=' * 78}")
    print("=== ZERO-SHOT BASE MODEL, ALL TARGETS ===")
    print("Affinity percentile: LOW = drug ranks above the decoy field (good).")
    print("NLL percentile:      LOW = drug's structure is unusually familiar to the base model.")
    print(f"\n{'target':9s} {'drug':12s} {'real':>5s} {'g.score':>8s} {'g.%ile':>7s} "
          f"{'d.score':>8s} {'d.%ile':>7s} {'nll':>7s} {'nll%ile':>8s} {'beats':>6s}")
    for t, r in results.items():
        ge = r["greedy"]["enrichment"]
        de = r["digit_exp"]["enrichment"]
        npct = r["smiles_nll"]["percentile"]
        nll = r["smiles_nll"]["reference"]
        dref = r["digit_exp"]["reference"]
        g_pct = f"{ge['percentile']:.1f}" if ge else "n/a"
        d_pct = f"{de['percentile']:.1f}" if de else "n/a"
        d_score = f"{dref:.1f}" if dref is not None else "n/a"
        nll_s = f"{nll:.3f}" if nll is not None else "n/a"
        nll_pct = f"{npct['percentile']:.1f}" if npct else "n/a"
        print(f"{t:9s} {r['reference_name'][:12]:12s} {r['reference_real']:>5} "
              f"{str(r['greedy']['reference']):>8s} {g_pct:>7s} "
              f"{d_score:>8s} {d_pct:>7s} {nll_s:>7s} {nll_pct:>8s} "
              f"{str(r['digit_exp']['beats_hard']):>6s}")

    print("\nINTERPRETATION GUIDE (fill in against try11's fine-tuned percentiles):")
    print("  try11 fine-tuned digit_exp enrichment was gaucher 0.2/0.2/0.8, attr ~50.8.")
    print("  - ambroxol low zero-shot + tafamidis high -> prior familiarity IS the cause.")
    print("  - both high zero-shot -> familiarity is NOT the cause; the difference was")
    print("    made by fine-tuning, and the search moves to what training did differently.")
    print("  - tafamidis low zero-shot but high after fine-tuning -> training destroyed a")
    print("    working prior, which is a damage problem rather than a knowledge gap.")
