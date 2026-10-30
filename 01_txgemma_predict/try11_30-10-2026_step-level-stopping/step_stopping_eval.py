"""
Step-level early stopping + dual readout, built on try09's leak-free harness.

WHY THIS EXISTS
---------------
All 4 completed try09 runs reported best_epoch = 0, with validation loss rising
monotonically afterwards (TTR seed0: 2.37 -> 3.10 -> 3.48 -> 3.56 -> 3.77). But at that
same checkpoint training loss was ~1.0 against validation ~2.37 -- the model had ALREADY
overfit inside epoch 1. So the true optimum is earlier than the first checkpoint, and
epoch-level evaluation is too coarse to find it.

This run makes the optimum visible and selects on it:

  1. Validation every N STEPS, not per epoch. N is set so there are ~7 evaluations per
     epoch, putting several measurement points inside epoch 1 where the minimum lives.
  2. Lower learning rate (default 1/3 of try09's 2e-4), which spreads the optimum over
     more steps so it is catchable rather than falling between measurements.
  3. N_EPOCH = 2. Not 1 -- at least one point after the minimum is needed to show it IS
     a minimum rather than just the last thing measured.
  4. The selected checkpoint is the lowest validation loss, never the last.
  5. The full validation curve is saved so the minimum is visible rather than asserted.

KEPT UNCHANGED FROM try09, so this run stays comparable to it
-------------------------------------------------------------
  - hard negatives held out of training BY SCAFFOLD (not just the 3 molecules, since a
    near-copy sharing the skeleton would otherwise let the model recall the label anyway)
  - scaffold-split validation set, so held-out loss measures generalization to unseen
    frameworks rather than near-neighbour recall
  - 500-decoy enrichment percentile as the category-level gate
  - 3 seeds, reported as a distribution

TWO READOUTS, SCORED SEPARATELY AND DELIBERATELY
------------------------------------------------
try10 established that greedy decoding returns the MODAL output rather than an estimate,
and that a digit-logit expectation fixes it (Gaucher's most-common value fell from 51% of
novel compounds to 5.5%). But the training change and the readout change must stay
attributable, so this run does NOT switch the readout -- it scores every resulting adapter
BOTH ways and reports them side by side.

Both readouts come from ONE generate call per molecule (output_scores=True), so greedy and
digit-expectation derive from the identical forward pass and cannot diverge for any reason
other than the readout itself.

ALSO FIXED HERE: ADAPTERS ARE SAVED
-----------------------------------
try09 wrote summaries but never called save_pretrained, so its epoch-0 models were lost and
try10 had to run on the older over-trained weights instead. Every run here saves its
adapter, before any scoring.

SCOPE
-----
One target per job via an explicit TARGET env var, so the try05 scope-coupling bug cannot
recur. 4 targets x 3 seeds = 12 runs across 4 jobs.

Measured split sizes, which set the step counts:
    fabry     89 comps ->  71 after holdout ->  ~60 train (300 lines)  /  ~11 val
    attr     493       -> 476              -> ~404       (2020)        /  ~72
    gaucher 1208       -> 1202             -> ~1021      (5105)        / ~181
    cf      1450       -> 1424             -> ~1210      (6050)        / ~214

Note on fabry: ~60 training and ~11 validation compounds is very small, so its validation
curve will be noisy and its selected step is weak evidence. Recorded here rather than
discovered afterwards.
"""
import json
import os
import random
import re
from collections import defaultdict

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

RDLogger.DisableLog("rdApp.*")

MODEL = "google/txgemma-9b-predict"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
OUT_ROOT = os.path.join(DATA_DIR, "step_stopping")

TARGET = os.environ.get("TARGET")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
N_EPOCH = int(os.environ.get("N_EPOCH", "2"))
# try09 used 2e-4 and its optimum landed before the first measurement. Default is 1/3 of
# that. If the minimum STILL lands at the first eval step, 1/10 (2e-5) is the follow-up --
# readable directly off the saved curve.
LR = float(os.environ.get("LR", str(2e-4 / 3)))
EVALS_PER_EPOCH = int(os.environ.get("EVALS_PER_EPOCH", "7"))
# Validation forwards cost the same as training forwards, so frequent evaluation uses a
# FIXED seeded subsample -- identical at every step, so the curve is comparable step to
# step. The full val set is measured once at the selected checkpoint.
VAL_EVAL_LINES = int(os.environ.get("VAL_EVAL_LINES", "400"))
N_DECOYS = int(os.environ.get("N_DECOYS", "500"))
GRAD_ACCUM = 8
VAL_FRACTION = 0.15
ALL_TARGETS = ["gaucher", "cf", "fabry", "attr"]

LORA_CONFIG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


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


def load_grouped(target):
    groups = defaultdict(list)
    for line in open(os.path.join(DATA_DIR, target, "train_blind.jsonl")):
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
    ii, ll = [], []
    for ex in examples:
        p = tok(ex["prompt"], add_special_tokens=True)["input_ids"]
        a = tok(ex["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ii.append(p + a)
        ll.append([-100] * len(p) + a)
    maxlen = max(len(x) for x in ii)
    input_ids = torch.full((len(examples), maxlen), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(examples), maxlen), dtype=torch.long)
    for i, (a, b) in enumerate(zip(ii, ll)):
        input_ids[i, :len(a)] = torch.tensor(a)
        labels[i, :len(b)] = torch.tensor(b)
        attn[i, :len(a)] = 1
    return input_ids.to(device), attn.to(device), labels.to(device)


@torch.no_grad()
def val_loss(model, tok, examples, device):
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i in range(len(examples)):
        ids, attn, labels = make_batch(tok, examples[i:i + 1], device)
        total += model(input_ids=ids, attention_mask=attn, labels=labels).loss.item()
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


def lora_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def train_step_level(model, tok, train_ex, val_eval_ex, n_epoch, lr, evals_per_epoch):
    """Trains with validation every N optimizer steps, returning the weights at the lowest
    validation loss seen -- never the last.

    Step-level rather than epoch-level because try09 showed the optimum falls INSIDE the
    first epoch, where epoch-level evaluation has no measurement points at all.
    """
    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps_per_epoch = max(1, len(train_ex) // GRAD_ACCUM)
    eval_every = max(1, steps_per_epoch // evals_per_epoch)
    print(f"  {len(train_ex)} train lines -> ~{steps_per_epoch} optimizer steps/epoch; "
          f"evaluating every {eval_every} steps "
          f"(~{max(1, steps_per_epoch // eval_every)} per epoch), lr={lr:.2e}", flush=True)

    # Loss before any update: the baseline the curve has to beat to mean anything. Also
    # the honest zero-shot reference point for this validation set.
    v0 = val_loss(model, tok, val_eval_ex, device)
    curve = [{"step": 0, "epoch": -1, "train_loss": None, "val_loss": v0}]
    best = {"step": 0, "val": v0, "state": lora_state(model), "epoch": -1}
    print(f"    step 0 (pre-training): val {v0:.4f}", flush=True)

    model.train()
    gstep = 0
    for epoch in range(n_epoch):
        random.shuffle(train_ex)
        run_loss, micro = 0.0, 0
        opt.zero_grad()
        for i in range(len(train_ex)):
            ids, attn, labels = make_batch(tok, train_ex[i:i + 1], device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / GRAD_ACCUM).backward()
            run_loss += out.loss.item()
            micro += 1
            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                gstep += 1
                if gstep % eval_every == 0:
                    tr = run_loss / micro
                    vl = val_loss(model, tok, val_eval_ex, device)
                    curve.append({"step": gstep, "epoch": epoch, "train_loss": tr,
                                  "val_loss": vl})
                    flag = ""
                    if vl < best["val"]:
                        best = {"step": gstep, "val": vl, "state": lora_state(model),
                                "epoch": epoch}
                        flag = "  <- best"
                    print(f"    step {gstep} (epoch {epoch}): train {tr:.4f}  "
                          f"val {vl:.4f}{flag}", flush=True)
        opt.step()
        opt.zero_grad()

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    model.eval()
    print(f"  SELECTED step {best['step']} (epoch {best['epoch']}), val {best['val']:.4f}, "
          f"from {len(curve)} measurements", flush=True)
    return curve, best["step"], best["val"], best["epoch"]


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
    """Greedy AND digit-logit expectation from ONE generate call.

    One call with output_scores=True guarantees both readouts come from the identical
    forward pass, so they cannot differ for any reason except the readout. Digit width is
    taken from the greedy decode, because the scale runs to 1000 = four digits and a fixed
    width of 3 would misread it by ~900 (the bug caught in try10 before it ran).
    """
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

    hard_scaffolds = {scaffold_of(h["smiles"]) for h in hard}
    hard_scaffolds.discard(None)
    scaffolds = {c: scaffold_of(c) for c in groups}
    removed = {c for c, s in scaffolds.items() if s in hard_scaffolds}
    trainable = {c: v for c, v in groups.items() if c not in removed}
    print(f"=== {TARGET}: {len(groups)} compounds; reference in training? "
          f"{ref_smiles in groups} (expect False)")
    print(f"hard negatives in training before holdout: "
          f"{sum(1 for h in hard if h['smiles'] in groups)}/{len(hard)}; after: "
          f"{sum(1 for h in hard if h['smiles'] in trainable)}/{len(hard)} (expect 0)")
    print(f"scaffold holdout removed {len(removed)} compounds "
          f"({100 * len(removed) / len(groups):.1f}%) -> {len(trainable)} trainable")

    decoys, pool_size = build_decoys(TARGET, N_DECOYS)
    print(f"decoys: {len(decoys)} from {pool_size} cross-target compounds")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dmap = digit_token_map(tok)
    print(f"per-digit tokenization: {'yes' if dmap else 'NO -- digit expectation disabled'}")

    results = {}
    for seed in SEEDS:
        tag = f"{TARGET}_seed{seed}"

        by_scaf = defaultdict(list)
        for c in trainable:
            by_scaf[scaffolds[c] or c].append(c)
        keys = sorted(by_scaf)
        random.Random(seed).shuffle(keys)
        want = max(1, int(VAL_FRACTION * len(trainable)))
        val_c, cur = [], 0
        for k in keys:
            if cur >= want:
                break
            val_c.extend(by_scaf[k])
            cur += len(by_scaf[k])
        vset = set(val_c)
        train_c = [c for c in trainable if c not in vset]
        train_ex = [ex for c in train_c for ex in trainable[c]]
        val_ex = [ex for c in val_c for ex in trainable[c]]
        val_eval_ex = (random.Random(1234).sample(val_ex, VAL_EVAL_LINES)
                       if len(val_ex) > VAL_EVAL_LINES else list(val_ex))
        print(f"\n=== {tag}: train {len(train_c)} compounds ({len(train_ex)} lines) | "
              f"val {len(val_c)} compounds ({len(val_ex)} lines, {len(val_eval_ex)} per "
              f"step-eval) | {len(keys)} scaffolds ===", flush=True)

        base = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="cuda:0", dtype=torch.bfloat16)
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        model = get_peft_model(base, LORA_CONFIG, adapter_name=tag)

        curve, best_step, best_val, best_epoch = train_step_level(
            model, tok, train_ex, val_eval_ex, N_EPOCH, LR, EVALS_PER_EPOCH)

        # try09 lost its models by never saving. Save before any scoring.
        adapter_path = os.path.join(out_dir, f"{tag}_adapter")
        model.save_pretrained(adapter_path, selected_adapters=[tag])
        print(f"  saved adapter -> {adapter_path}")

        # Full val-set loss at the selected checkpoint, for comparability with try09's
        # whole-val-set numbers (the curve above uses the fixed subsample).
        full_val = val_loss(model, tok, val_ex, next(model.parameters()).device)
        print(f"  full validation set loss at selected checkpoint: {full_val:.4f}")

        print(f"=== Scoring {tag}: both readouts, one forward pass each ===", flush=True)
        ref_g, ref_d = score_both(model, tok, prompt_for(template, ref_smiles), dmap)
        print(f"  reference: greedy={ref_g} "
              f"digit_exp={round(ref_d, 1) if ref_d is not None else None} "
              f"(real {eval_set['reference']['score']})")

        hard_scored = []
        for h in hard:
            g, d = score_both(model, tok, prompt_for(template, h["smiles"]), dmap)
            hard_scored.append({"name": h["name"], "real": h["real"],
                                "greedy": g, "digit_exp": d})
            print(f"  hard neg {h['name']} (real {h['real']}, held out): greedy={g} "
                  f"digit_exp={round(d, 1) if d is not None else None}")

        dg, dd = [], []
        for i, s in enumerate(decoys):
            g, d = score_both(model, tok, prompt_for(template, s), dmap)
            dg.append(g)
            dd.append(d)
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(decoys)} decoys", flush=True)

        res = {"seed": seed, "lr": LR, "n_epoch": N_EPOCH,
               "n_train_compounds": len(train_c), "n_val_compounds": len(val_c),
               "n_train_lines": len(train_ex), "validation_curve": curve,
               "selected_step": best_step, "selected_epoch": best_epoch,
               "selected_val_loss": best_val, "full_val_loss_at_selected": full_val,
               "adapter_path": adapter_path,
               "reference_real": eval_set["reference"]["score"]}
        for name, refv, dvals in (("greedy", ref_g, dg), ("digit_exp", ref_d, dd)):
            beats = (refv is not None and
                     all(h[name] is None or refv < h[name] for h in hard_scored))
            collapsed = (refv is not None and all(h[name] == refv for h in hard_scored))
            e = enrichment(refv, dvals)
            res[name] = {"reference": refv, "beats_hard": beats, "collapsed": collapsed,
                         "enrichment": e,
                         "n_distinct_decoys": len({round(v) for v in dvals if v is not None}),
                         "hard": [h[name] for h in hard_scored]}
            if e:
                print(f"  ENRICHMENT [{name}]: {e['rank']}/{e['n']} = "
                      f"{e['percentile']:.1f} percentile | beats_hard={beats} | "
                      f"distinct_decoys={res[name]['n_distinct_decoys']}")
        res["hard_detail"] = hard_scored
        results[tag] = res

        with open(os.path.join(out_dir, f"{tag}_decoy_scores.json"), "w") as f:
            json.dump({"decoys": decoys, "greedy": dg, "digit_exp": dd}, f)
        del model, base
        torch.cuda.empty_cache()
        with open(os.path.join(out_dir, "step_stopping_summary.json"), "w") as f:
            json.dump({"target": TARGET, "seeds": SEEDS, "lr": LR, "n_epoch": N_EPOCH,
                       "evals_per_epoch": EVALS_PER_EPOCH, "results": results}, f, indent=2)

    print(f"\n=== FINAL SUMMARY: {TARGET} (step-level stopping, lr={LR:.2e}, "
          f"{len(SEEDS)} seeds) ===")
    header = (f"{'run':16s} {'sel.step':>9s} {'sel.ep':>7s} {'val':>7s} "
              f"{'greedy%ile':>11s} {'digit%ile':>10s} {'g.beats':>8s} {'d.beats':>8s}")
    print(header)
    for tag, r in results.items():
        ge, de = r["greedy"]["enrichment"], r["digit_exp"]["enrichment"]
        gs = f"{ge['percentile']:.1f}" if ge else "n/a"
        ds = f"{de['percentile']:.1f}" if de else "n/a"
        print(f"{tag:16s} {r['selected_step']:>9} {r['selected_epoch']:>7} "
              f"{r['selected_val_loss']:>7.3f} {gs:>11s} {ds:>10s} "
              f"{str(r['greedy']['beats_hard']):>8s} "
              f"{str(r['digit_exp']['beats_hard']):>8s}")

    steps = [r["selected_step"] for r in results.values()]
    print(f"\nselected steps: {steps}")
    if steps and max(steps) == 0:
        print("  WARNING: every seed selected step 0, the PRE-TRAINING checkpoint. Training")
        print("  never improved held-out loss at all, so these adapters are effectively the")
        print("  base model. Lower LR further (2e-5) before concluding anything about")
        print("  fine-tuning -- or the task is not learnable from this data at this scale.")
    elif steps and min(steps) <= 1:
        print("  NOTE: a seed selected step <=1, so the optimum is still at the very start.")
        print("  Re-run at lr=2e-5 (1/10 of try09's) before trusting these adapters.")
    for name in ("greedy", "digit_exp"):
        ps = [r[name]["enrichment"]["percentile"] for r in results.values()
              if r[name]["enrichment"]]
        if ps:
            print(f"{name:10s} enrichment across seeds: {[round(p, 1) for p in ps]} "
                  f"(spread {max(ps) - min(ps):.1f})")
        nb = sum(1 for r in results.values() if r[name]["beats_hard"])
        print(f"{name:10s} leak-free hard-negative test: {nb}/{len(results)} seeds passed")
