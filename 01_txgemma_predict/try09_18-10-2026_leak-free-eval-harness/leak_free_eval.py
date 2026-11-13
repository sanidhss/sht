"""
Leak-free evaluation harness -- rebuilds the test that try07/try08 showed was broken.

WHY THIS EXISTS
---------------
try08 found that all 12 hard negatives, across all four targets, were present in their
training sets. Only the reference drug was held out. So the hard-negative check compared
a GENERALIZATION task (predict a held-out drug) against a MEMORIZATION task (recall three
trained analogs) -- not what it claimed to measure ("the test that actually indicates the
model learned the right structure-activity signal").

That single flaw explains most of the project's confusing results:
  - The "collapse" signature behind the Fabry and TTR diagnoses (one identical value for
    the reference and all three analogs) is the model applying a memorized neighbour's
    label to the whole structural family. Gaucher emits 791 -- a hard negative's real
    label -- for 51% of 200 diverse novel compounds.
  - Fabry's diastereomer/stereochemistry diagnosis, carried for weeks, is most likely an
    artifact of this rather than a finding about chiral centers.

try07 independently showed that single-run verdicts are noise-dominated: the same target,
data size, pipeline and hyperparameters, varying only which 493 compounds were drawn,
predicted the held-out reference at 785, 408 and 480 (real label 685; full-size run 691).
A 377-point swing, with seeds landing on opposite sides of pass/fail.

So this harness changes four things at once. Each is necessary; none is sufficient alone.

  1. HARD NEGATIVES HELD OUT OF TRAINING, by scaffold.
     Not just the three molecules -- every training compound sharing their Bemis-Murcko
     scaffold, otherwise near-identical compounds remain and "generalization" is still
     near-neighbour recall. Measured cost of this holdout: CF 1.8% of compounds, TTR 3.4%,
     Gaucher 0.5%, Fabry 20.2%.

  2. SCAFFOLD SPLIT for validation, not a random split.
     A random split leaves close analogs of held-out compounds in training. Splitting by
     scaffold forces generalization to unseen molecular frameworks -- which is exactly the
     gap try08 measured (held-in resolution 72-78% vs novel 23-49%).

  3. PER-EPOCH VALIDATION LOSS, and the best epoch is kept.
     There was previously NO validation set at all: training ran a fixed 3 epochs and
     never evaluated on held-out data, so nobody knew whether epoch 3 was past the
     overfitting point. Training loss ending at 0.26-0.37 suggested it was. With 5x SMILES
     augmentation the model sees each exact 3-digit label 15 times (5 variants x 3 epochs).
     This runs more epochs and keeps the weights at minimum validation loss.

  4. AN ENRICHMENT GATE, replacing the 5-molecule easy-negative test.
     The old easy-negative check used 5 fixed household drugs, and try07 showed those
     scores are unstable: across three seeds imatinib scored 704/315/357, caffeine
     952/652/952, metformin 632/911/653. Ambroxol beat metformin in 2 of 3 seeds, so
     "Gaucher loses to metformin" was one unlucky draw, not a property of the model. Those
     five molecules are also small and simple, far outside the training distribution, so
     the model returns near-arbitrary values for them.
     Replaced with: where does the reference rank among 500 diverse decoys? Percentile
     rank over hundreds of molecules is stable where 5 pairwise comparisons are not.

     Computed from try08's existing data as a preview of this metric (200 novel compounds,
     single seed): CF 11.5%, Fabry 15.5%, Gaucher 38.5%, TTR 47.7% (~chance). That split
     is why CF and TTR are the two targets run first -- the informative extremes.

  5. MULTIPLE SEEDS, reported as a distribution.
     The seed varies the scaffold split. A single number is not reportable given try07.

DECOY-SET CAVEAT, stated so the enrichment number is not over-read
------------------------------------------------------------------
There is no DrugBank or ZINC library on this system, and this script makes no network
calls (so the comparison cannot shift under it). Decoys are therefore drawn from the other
targets' training compounds -- molecules with measured bioactivity against a DIFFERENT
protein, excluded from this target's own data.

These are bioactive, drug-like molecules rather than random chemical matter, which makes
this a CONSERVATIVE (harder) decoy set than random decoys would be: some could legitimately
be potent here. A good percentile against them is therefore strong evidence; a poor one is
weaker evidence than it looks. A proper absolute enrichment number needs random decoys from
an external library, which is a separate piece of work.

SCOPE
-----
TARGET and SEEDS come from env vars so each target runs as its own job (and so the
scope-coupling bug that cost try05 a run cannot recur). One target, three seeds per job.
Writes to leak_free_eval/<target>/ and touches no previous try's outputs.
"""
import copy
import json
import os
import random
import re
from collections import defaultdict

import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

RDLogger.DisableLog("rdApp.*")

MODEL = "google/txgemma-9b-predict"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
OUT_ROOT = os.path.join(DATA_DIR, "leak_free_eval")

TARGET = os.environ.get("TARGET")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
N_EPOCH = int(os.environ.get("N_EPOCH", "5"))
N_DECOYS = int(os.environ.get("N_DECOYS", "500"))
VAL_FRACTION = 0.15
ALL_TARGETS = ["gaucher", "cf", "fabry", "attr"]

LORA_CONFIG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def canon(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m else None


def scaffold_of(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def load_grouped(target):
    """{canonical_smiles: [augmented training lines]} for one target.

    The jsonl holds only {prompt, answer}, so compound identity is recovered from the
    prompt's SMILES line and canonicalized -- which also regroups the 5 augmented
    variants of each compound.
    """
    groups = defaultdict(list)
    path = os.path.join(DATA_DIR, target, "train_blind.jsonl")
    for line in open(path):
        d = json.loads(line)
        m = re.search(r"Drug SMILES: (.+)", d["prompt"])
        if not m:
            continue
        c = canon(m.group(1).strip())
        if c:
            groups[c].append(d)
    return groups


def prompt_for(template, smiles):
    new, n = re.subn(r"(Drug SMILES: ).+", lambda m: m.group(1) + smiles, template, count=1)
    if n != 1:
        raise ValueError("SMILES substitution failed")
    return new


def make_batch(tok, examples, device):
    input_ids_list, labels_list = [], []
    for ex in examples:
        prompt_ids = tok(ex["prompt"], add_special_tokens=True)["input_ids"]
        answer_ids = tok(ex["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        input_ids_list.append(ids)
        labels_list.append(labels)
    maxlen = max(len(x) for x in input_ids_list)
    pad_id = tok.pad_token_id
    input_ids = torch.full((len(examples), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(examples), maxlen), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        attn[i, :len(ids)] = 1
    return input_ids.to(device), attn.to(device), labels.to(device)


@torch.no_grad()
def val_loss(model, tok, examples, device):
    """Mean held-out loss. The thing that was missing entirely before."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(examples), 1):
        ids, attn, labels = make_batch(tok, examples[i:i + 1], device)
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
        total += out.loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def lora_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if "lora_" in k}


def train_with_validation(model, tok, train_ex, val_ex, n_epoch, lr=2e-4, grad_accum=8):
    """Trains, measuring held-out loss each epoch, and returns the weights at MINIMUM
    validation loss rather than whatever the last epoch produced."""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history, best = [], {"epoch": None, "val": float("inf"), "state": None}
    model.train()
    for epoch in range(n_epoch):
        random.shuffle(train_ex)
        total, step = 0.0, 0
        opt.zero_grad()
        for i in range(0, len(train_ex), 1):
            ids, attn, labels = make_batch(tok, train_ex[i:i + 1], device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / grad_accum).backward()
            total += out.loss.item()
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        tr = total / max(step, 1)
        vl = val_loss(model, tok, val_ex, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": vl})
        flag = ""
        if vl < best["val"]:
            best = {"epoch": epoch, "val": vl, "state": lora_state(model)}
            flag = "  <- best so far"
        print(f"  epoch {epoch}: train {tr:.4f}  val {vl:.4f}{flag}", flush=True)
    if best["state"] is not None:
        missing = model.load_state_dict(best["state"], strict=False)
        del missing
    model.eval()
    print(f"  restored weights from epoch {best['epoch']} (val {best['val']:.4f})")
    return history, best["epoch"], best["val"]


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


def build_decoys(target, n):
    """Diverse decoys from the OTHER targets' compounds, excluding anything in this
    target's own data. Built once per target and reused across seeds, so the enrichment
    numbers are comparable between seeds."""
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
    picked = [mols[i][0] for i in idx]
    sims = []
    for i in range(min(len(fps), 150)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:min(len(fps), 150)]))
    return picked, (sum(sims) / len(sims) if sims else float("nan")), len(pool)


if __name__ == "__main__":
    if TARGET not in ALL_TARGETS:
        raise SystemExit(f"set TARGET to one of {ALL_TARGETS} (got {TARGET!r})")
    out_dir = os.path.join(OUT_ROOT, TARGET)
    os.makedirs(out_dir, exist_ok=True)

    groups = load_grouped(TARGET)
    eval_set = json.load(open(os.path.join(DATA_DIR, TARGET, "eval_set.json")))
    template = eval_set["reference"]["prompt"]
    ref_smiles = canon(eval_set["reference"]["smiles"])
    hard = [{"smiles": canon(c["smiles"]), "real": c["score"],
             "name": c.get("name") or c["smiles"][:20]} for c in eval_set["hard_negatives"]]

    print(f"=== {TARGET}: {len(groups)} compounds in training file ===")
    print(f"reference '{eval_set['reference']['name']}' in training file? "
          f"{ref_smiles in groups}  (expected False)")

    # --- FIX 1: hold out hard negatives AND their scaffolds -------------------
    hard_scaffolds = {scaffold_of(h["smiles"]) for h in hard}
    hard_scaffolds.discard(None)
    scaffolds = {c: scaffold_of(c) for c in groups}
    leaked = [h["smiles"] for h in hard if h["smiles"] in groups]
    removed = [c for c, s in scaffolds.items() if s in hard_scaffolds]
    print(f"hard negatives found in training file: {len(leaked)}/{len(hard)} "
          f"(this is the leak try08 found)")
    print(f"holding out {len(hard_scaffolds)} hard-negative scaffold(s) -> removing "
          f"{len(removed)} compounds ({100*len(removed)/len(groups):.1f}%)")
    trainable = {c: v for c, v in groups.items() if c not in set(removed)}

    decoys, decoy_tan, pool_size = build_decoys(TARGET, N_DECOYS)
    print(f"decoys: {len(decoys)} picked from {pool_size} cross-target compounds, "
          f"mean pairwise Tanimoto {decoy_tan:.3f}")
    with open(os.path.join(out_dir, "decoys.json"), "w") as f:
        json.dump({"n": len(decoys), "mean_pairwise_tanimoto": decoy_tan,
                   "pool_size": pool_size, "decoys": decoys}, f, indent=2)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    results = {}
    for seed in SEEDS:
        tag = f"{TARGET}_seed{seed}"

        # --- FIX 2: scaffold split for validation ----------------------------
        by_scaf = defaultdict(list)
        for c in trainable:
            by_scaf[scaffolds[c] or c].append(c)
        scaf_keys = sorted(by_scaf)
        random.Random(seed).shuffle(scaf_keys)
        n_val_target = max(1, int(VAL_FRACTION * len(trainable)))
        val_c, cur = [], 0
        for k in scaf_keys:
            if cur >= n_val_target:
                break
            val_c.extend(by_scaf[k])
            cur += len(by_scaf[k])
        val_set = set(val_c)
        train_c = [c for c in trainable if c not in val_set]
        train_ex = [ex for c in train_c for ex in trainable[c]]
        val_ex = [ex for c in val_c for ex in trainable[c]]
        print(f"\n=== {tag}: scaffold split -> train {len(train_c)} compounds "
              f"({len(train_ex)} lines) | val {len(val_c)} compounds ({len(val_ex)} lines) "
              f"across {len(scaf_keys)} scaffolds ===", flush=True)

        print(f"=== Loading base model for {tag} ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="cuda:0", dtype=torch.bfloat16)
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        model = get_peft_model(base, LORA_CONFIG, adapter_name=tag)

        # --- FIX 3: per-epoch validation loss, keep best epoch ---------------
        print(f"=== Training {tag} ({N_EPOCH} epochs max, keeping best val) ===", flush=True)
        history, best_epoch, best_val = train_with_validation(
            model, tok, train_ex, val_ex, N_EPOCH)

        print(f"=== Evaluating {tag} (leak-free) ===", flush=True)
        ref_pred = to_int(score(model, tok, prompt_for(template, ref_smiles)))
        hard_preds = []
        for h in hard:
            p = to_int(score(model, tok, prompt_for(template, h["smiles"])))
            hard_preds.append({"name": h["name"], "real": h["real"], "pred": p})
            print(f"  hard neg {h['name']} (real {h['real']}, NOW HELD OUT): {p}")
        print(f"  reference predicted {ref_pred} (real {eval_set['reference']['score']})")

        beats_hard = (ref_pred is not None and
                      all(h["pred"] is None or ref_pred < h["pred"] for h in hard_preds))
        collapsed = (ref_pred is not None and
                     all(h["pred"] == ref_pred for h in hard_preds))

        # --- FIX 4: enrichment gate ------------------------------------------
        print(f"  scoring {len(decoys)} decoys for enrichment...", flush=True)
        decoy_vals = []
        for s in decoys:
            v = to_int(score(model, tok, prompt_for(template, s)))
            if v is not None:
                decoy_vals.append(v)
        better = sum(1 for v in decoy_vals if v < ref_pred) if ref_pred is not None else None
        pct = (100.0 * better / len(decoy_vals)) if decoy_vals and better is not None else None
        n_distinct = len(set(decoy_vals))
        top_vals = sorted(((decoy_vals.count(v), v) for v in set(decoy_vals)), reverse=True)[:5]
        print(f"  ENRICHMENT: reference ranks {better}/{len(decoy_vals)} "
              f"= {pct:.1f} percentile (lower is better; 50 = chance)")
        print(f"  decoy output diversity: {n_distinct} distinct values; "
              f"most common {[(v, c) for c, v in top_vals]}")

        results[tag] = {
            "seed": seed, "n_train_compounds": len(train_c), "n_val_compounds": len(val_c),
            "history": history, "best_epoch": best_epoch, "best_val_loss": best_val,
            "reference_pred": ref_pred, "reference_real": eval_set["reference"]["score"],
            "hard_negatives": hard_preds, "beats_hard_leakfree": beats_hard,
            "collapsed": collapsed, "enrichment_percentile": pct,
            "enrichment_rank": better, "n_decoys_scored": len(decoy_vals),
            "decoy_distinct_values": n_distinct,
            "decoy_top_values": [(v, c) for c, v in top_vals],
        }

        del model, base
        torch.cuda.empty_cache()
        with open(os.path.join(out_dir, "leakfree_summary.json"), "w") as f:
            json.dump({"target": TARGET, "seeds": SEEDS, "n_epoch_max": N_EPOCH,
                       "decoy_mean_pairwise_tanimoto": decoy_tan,
                       "n_hard_negatives_removed_from_training": len(removed),
                       "results": results}, f, indent=2)

    print(f"\n=== FINAL SUMMARY: {TARGET} (leak-free, scaffold-split, {len(SEEDS)} seeds) ===")
    for tag, r in results.items():
        print(f"{tag}: ref={r['reference_pred']} "
              f"hard={[h['pred'] for h in r['hard_negatives']]} "
              f"beats_hard={r['beats_hard_leakfree']} collapsed={r['collapsed']} "
              f"enrichment={r['enrichment_percentile']}%ile "
              f"best_epoch={r['best_epoch']} val={r['best_val_loss']:.4f}")
    preds = [r["reference_pred"] for r in results.values() if r["reference_pred"] is not None]
    pcts = [r["enrichment_percentile"] for r in results.values()
            if r["enrichment_percentile"] is not None]
    if preds:
        print(f"\nreference prediction across seeds: {preds} "
              f"(spread {max(preds)-min(preds)})")
    if pcts:
        print(f"enrichment percentile across seeds: {[round(p,1) for p in pcts]} "
              f"(spread {max(pcts)-min(pcts):.1f})")
    n_pass = sum(1 for r in results.values() if r["beats_hard_leakfree"])
    print(f"leak-free hard-negative test: {n_pass}/{len(results)} seeds passed")
    print("\nRead against try08's leaky numbers for the same target, not against zero:")
    print("  a target that passed only because its analogs were memorized should now fail")
    print("  a target that still passes is showing real within-family discrimination")
    print("  enrichment percentile is the category gate: <10 strong, ~50 = chance")
