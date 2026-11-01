"""
Fabry re-run: cross-validation so all 71 compounds inform the stopping decision.

WHY
---
try11 gave Fabry an inconclusive result. The cause was not the method -- it was that Fabry
got only ELEVEN molecules in its checking set, while Gaucher got 180-208:

    Gaucher  1208 compounds -> 6 removed by scaffold holdout (0.5%) -> 1202 -> 15% = 180-208
    Fabry      89 compounds -> 18 removed (20.2%)                  ->   71 -> 15% =  11

Two things compounded. A fixed 15% validation fraction is fine at 1,200 compounds and
useless at 71. And Fabry loses 20% to the hard-negative scaffold holdout versus Gaucher's
0.5%, because Fabry's compounds are nearly all sugar-shaped iminosugars that share skeletons
with migalastat's diastereomers.

Eleven molecules cannot resolve a validation curve, which is why try11's three Fabry seeds
disagreed wildly (enrichment 20.0 / 58.4 / 27.2) while Gaucher's agreed closely (0.2 / 0.2 /
0.8).

For the record, Fabry has never produced a positive result: try04 failed, try05 failed (the
436-for-everything collapse), try08's apparent top-15.5% was proven in try10 to be a
greedy-decoding artifact (honest scoring put it at 83rd percentile, worse than chance), and
try11 was inconclusive. This run is about whether that verdict is REAL or just unmeasurable,
not about rescuing a result that previously worked.

WHAT CHANGES -- ONLY THE VALIDATION SCHEME
------------------------------------------
Everything that worked for Gaucher is held identical: lr=6.67e-5, N_EPOCH=2, ~7 validation
measurements per epoch, scaffold-held hard negatives, 500-decoy enrichment, greedy AND
digit-logit readouts from one forward pass, adapters saved.

The only change is how the stopping step is chosen:

  PHASE 1 -- 6-fold scaffold cross-validation.
    The 26 scaffold groups are split into 6 folds (greedy balancing by compound count, since
    group sizes are uneven: 10, 6, 4, 4, 4, 3, ...). Each fold trains on ~60 compounds and
    validates on the ~11 it held out, recording a validation curve. No scoring in this phase.
    Across the 6 folds, ALL 71 compounds serve as a validation molecule exactly once, so the
    averaged curve is built from 71 measurements instead of 11.

  PHASE 2 -- train on all 71 and score.
    3 seeds. Each trains on all 71 compounds, stopping at the optimum found in phase 1, then
    scores the reference, the 3 held-out hard negatives and the 500 decoys.

The optimum transfers between phases as a FRACTION OF AN EPOCH, not an absolute step count:
phase 1 trains on ~60 compounds (~37 steps/epoch) and phase 2 on 71 (~44 steps/epoch), so
step 15 of 37 means 0.405 epochs, which is step ~18 in phase 2. Carrying the raw step number
across would stop phase 2 roughly 20% too early.

The reference drug and the 3 hard negatives are excluded before any folding, so they are held
out of every fold and of the final models. The 500 decoys are identical across folds and
seeds, so enrichment stays comparable to try11's Gaucher numbers.
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
OUT_DIR = os.path.join(DATA_DIR, "fabry_crossval")
TARGET = "fabry"
ALL_TARGETS = ["gaucher", "cf", "fabry", "attr"]

K_FOLDS = int(os.environ.get("K_FOLDS", "6"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
N_EPOCH = int(os.environ.get("N_EPOCH", "2"))
LR = float(os.environ.get("LR", str(2e-4 / 3)))
EVALS_PER_EPOCH = int(os.environ.get("EVALS_PER_EPOCH", "7"))
N_DECOYS = int(os.environ.get("N_DECOYS", "500"))
GRAD_ACCUM = 8

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
    was = model.training
    model.eval()
    total = 0.0
    for i in range(len(examples)):
        ids, attn, labels = make_batch(tok, examples[i:i + 1], device)
        total += model(input_ids=ids, attention_mask=attn, labels=labels).loss.item()
    if was:
        model.train()
    return total / max(len(examples), 1)


def balanced_folds(scaf_groups, k, seed=0):
    """Split scaffold groups into k folds balanced by COMPOUND count.

    Group sizes are very uneven here (10, 6, 4, 4, 4, 3, ...), so round-robin on groups
    would give folds differing by ~3x in molecule count. Largest-group-first into the
    currently-smallest fold keeps them comparable, which matters because each fold's curve
    is weighted equally in the average.
    """
    keys = sorted(scaf_groups, key=lambda s: (-len(scaf_groups[s]), s))
    rng = random.Random(seed)
    rng.shuffle(keys)
    keys.sort(key=lambda s: -len(scaf_groups[s]))
    folds = [[] for _ in range(k)]
    for key in keys:
        smallest = min(range(k), key=lambda i: len(folds[i]))
        folds[smallest].extend(scaf_groups[key])
    return folds


def train_to_curve(model, tok, train_ex, val_ex, n_epoch, lr, evals_per_epoch):
    """Train, recording a validation curve. Returns the curve only -- phase 1 does not
    need the weights, just the shape of the overfitting."""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps_per_epoch = max(1, len(train_ex) // GRAD_ACCUM)
    eval_every = max(1, steps_per_epoch // evals_per_epoch)
    v0 = val_loss(model, tok, val_ex, device)
    curve = [{"step": 0, "epoch_frac": 0.0, "val_loss": v0}]
    print(f"    {len(train_ex)} lines, {steps_per_epoch} steps/epoch, eval every "
          f"{eval_every}; step 0 val {v0:.4f}", flush=True)
    model.train()
    gstep = 0
    for epoch in range(n_epoch):
        random.shuffle(train_ex)
        micro = 0
        opt.zero_grad()
        for i in range(len(train_ex)):
            ids, attn, labels = make_batch(tok, train_ex[i:i + 1], device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / GRAD_ACCUM).backward()
            micro += 1
            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                gstep += 1
                if gstep % eval_every == 0:
                    vl = val_loss(model, tok, val_ex, device)
                    frac = gstep / steps_per_epoch
                    curve.append({"step": gstep, "epoch_frac": frac, "val_loss": vl})
                    print(f"    step {gstep} ({frac:.3f} epochs): val {vl:.4f}", flush=True)
        opt.step()
        opt.zero_grad()
    return curve, steps_per_epoch


def train_fixed_steps(model, tok, train_ex, stop_step, lr):
    """Phase 2: train for exactly stop_step optimizer steps on the full dataset."""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    gstep, micro, done = 0, 0, False
    for epoch in range(N_EPOCH):
        if done:
            break
        random.shuffle(train_ex)
        opt.zero_grad()
        for i in range(len(train_ex)):
            ids, attn, labels = make_batch(tok, train_ex[i:i + 1], device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / GRAD_ACCUM).backward()
            micro += 1
            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                gstep += 1
                if gstep >= stop_step:
                    done = True
                    break
    model.eval()
    print(f"  trained {gstep} steps (target {stop_step})")
    return gstep


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


def fresh_model(tok_unused=None):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="cuda:0", dtype=torch.bfloat16)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    return base


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

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
    print(f"=== fabry: {len(groups)} compounds; reference in training? "
          f"{ref_smiles in groups} (expect False)")
    print(f"hard negatives in training before holdout: "
          f"{sum(1 for h in hard if h['smiles'] in groups)}/3; after: "
          f"{sum(1 for h in hard if h['smiles'] in trainable)}/3 (expect 0)")
    print(f"scaffold holdout removed {len(removed)} ({100*len(removed)/len(groups):.1f}%) "
          f"-> {len(trainable)} trainable compounds")

    scaf_groups = defaultdict(list)
    for c in trainable:
        scaf_groups[scaffolds[c] or c].append(c)
    folds = balanced_folds(scaf_groups, K_FOLDS)
    print(f"{len(scaf_groups)} scaffold groups -> {K_FOLDS} folds, sizes "
          f"{[len(f) for f in folds]} compounds (total {sum(len(f) for f in folds)})")
    assert sum(len(f) for f in folds) == len(trainable), "folds must partition the data"

    decoys, pool_size = build_decoys(TARGET, N_DECOYS)
    print(f"decoys: {len(decoys)} from {pool_size} cross-target compounds")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dmap = digit_token_map(tok)
    print(f"per-digit tokenization: {'yes' if dmap else 'NO'}")

    # ---------------- PHASE 1: cross-validated stopping point ----------------
    print(f"\n########## PHASE 1: {K_FOLDS}-fold scaffold cross-validation ##########")
    print("All 71 compounds serve as a validation molecule exactly once across the folds,")
    print("so the averaged curve rests on 71 measurements rather than 11.\n")
    fold_curves = []
    for fi, fold in enumerate(folds):
        vs = set(fold)
        tr_c = [c for c in trainable if c not in vs]
        train_ex = [ex for c in tr_c for ex in trainable[c]]
        # One variant per validation compound: maximum distinct molecules per unit compute.
        val_ex = [trainable[c][0] for c in fold]
        print(f"=== fold {fi}: train {len(tr_c)} compounds ({len(train_ex)} lines) | "
              f"validate {len(fold)} compounds ({len(val_ex)} molecules) ===", flush=True)
        base = fresh_model()
        model = get_peft_model(base, LORA_CONFIG, adapter_name=f"fold{fi}")
        curve, spe = train_to_curve(model, tok, train_ex, val_ex, N_EPOCH, LR,
                                    EVALS_PER_EPOCH)
        fold_curves.append({"fold": fi, "n_val": len(fold), "n_train": len(tr_c),
                            "steps_per_epoch": spe, "curve": curve})
        del model, base
        torch.cuda.empty_cache()
        with open(os.path.join(OUT_DIR, "phase1_folds.json"), "w") as f:
            json.dump(fold_curves, f, indent=2)

    # Average the folds on a common x-axis of epoch fraction, since folds differ slightly
    # in steps/epoch.
    by_frac = defaultdict(list)
    for fc in fold_curves:
        for p in fc["curve"]:
            by_frac[round(p["epoch_frac"], 2)].append(p["val_loss"])
    avg = sorted(({"epoch_frac": k, "mean_val": sum(v) / len(v), "n_folds": len(v)}
                  for k, v in by_frac.items()), key=lambda r: r["epoch_frac"])
    print("\n--- averaged validation curve (all 71 molecules contribute) ---")
    for r in avg:
        print(f"  {r['epoch_frac']:.2f} epochs: mean val {r['mean_val']:.4f} "
              f"({r['n_folds']} folds)")
    # Only consider points measured by a majority of folds, so a single fold's stray
    # measurement cannot define the optimum.
    usable = [r for r in avg if r["n_folds"] >= max(2, K_FOLDS // 2)]
    best = min(usable, key=lambda r: r["mean_val"])
    print(f"\nCHOSEN STOPPING POINT: {best['epoch_frac']:.3f} epochs "
          f"(mean val {best['mean_val']:.4f}, from {best['n_folds']} folds)")

    # ---------------- PHASE 2: train on all 71, then score ----------------
    print(f"\n########## PHASE 2: train on all {len(trainable)} compounds, "
          f"{len(SEEDS)} seeds ##########")
    full_ex_template = [ex for c in trainable for ex in trainable[c]]
    full_spe = max(1, len(full_ex_template) // GRAD_ACCUM)
    stop_step = max(1, round(best["epoch_frac"] * full_spe))
    print(f"full set: {len(trainable)} compounds, {len(full_ex_template)} lines, "
          f"{full_spe} steps/epoch -> stopping at step {stop_step} "
          f"({best['epoch_frac']:.3f} epochs)\n")

    results = {}
    for seed in SEEDS:
        tag = f"fabry_cv_seed{seed}"
        random.seed(seed)
        train_ex = list(full_ex_template)
        print(f"=== {tag} ===", flush=True)
        base = fresh_model()
        model = get_peft_model(base, LORA_CONFIG, adapter_name=tag)
        actual = train_fixed_steps(model, tok, train_ex, stop_step, LR)
        adapter_path = os.path.join(OUT_DIR, f"{tag}_adapter")
        model.save_pretrained(adapter_path, selected_adapters=[tag])

        ref_g, ref_d = score_both(model, tok, prompt_for(template, ref_smiles), dmap)
        print(f"  reference migalastat: greedy={ref_g} "
              f"digit_exp={round(ref_d,1) if ref_d is not None else None} "
              f"(real {eval_set['reference']['score']})")
        hard_scored = []
        for h in hard:
            g, d = score_both(model, tok, prompt_for(template, h["smiles"]), dmap)
            hard_scored.append({"name": h["name"], "real": h["real"],
                                "greedy": g, "digit_exp": d})
            print(f"  hard neg {h['name']} (real {h['real']}, held out): greedy={g} "
                  f"digit_exp={round(d,1) if d is not None else None}")
        dg, dd = [], []
        for i, s in enumerate(decoys):
            g, d = score_both(model, tok, prompt_for(template, s), dmap)
            dg.append(g)
            dd.append(d)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(decoys)} decoys", flush=True)

        res = {"seed": seed, "stop_step": stop_step, "actual_steps": actual,
                "chosen_epoch_frac": best["epoch_frac"],
                "n_train_compounds": len(trainable), "adapter_path": adapter_path,
                "reference_real": eval_set["reference"]["score"]}
        for name, refv, dvals in (("greedy", ref_g, dg), ("digit_exp", ref_d, dd)):
            beats = (refv is not None and
                     all(h[name] is None or refv < h[name] for h in hard_scored))
            collapsed = (refv is not None and all(h[name] == refv for h in hard_scored))
            e = enrichment(refv, dvals)
            res[name] = {"reference": refv, "beats_hard": beats, "collapsed": collapsed,
                         "enrichment": e,
                         "n_distinct_decoys": len({round(v) for v in dvals if v is not None})}
            if e:
                print(f"  ENRICHMENT [{name}]: {e['rank']}/{e['n']} = "
                      f"{e['percentile']:.1f} percentile | beats_hard={beats} | "
                      f"distinct_decoys={res[name]['n_distinct_decoys']}")
        res["hard_detail"] = hard_scored
        results[tag] = res
        with open(os.path.join(OUT_DIR, f"{tag}_decoy_scores.json"), "w") as f:
            json.dump({"decoys": decoys, "greedy": dg, "digit_exp": dd}, f)
        del model, base
        torch.cuda.empty_cache()
        with open(os.path.join(OUT_DIR, "fabry_crossval_summary.json"), "w") as f:
            json.dump({"k_folds": K_FOLDS, "lr": LR, "n_epoch": N_EPOCH,
                       "chosen_epoch_frac": best["epoch_frac"],
                       "averaged_curve": avg, "phase1_folds": fold_curves,
                       "stop_step_full": stop_step, "results": results}, f, indent=2)

    print("\n=== FINAL SUMMARY: fabry cross-validated ===")
    print(f"stopping point from {K_FOLDS}-fold CV on all {len(trainable)} compounds: "
          f"{best['epoch_frac']:.3f} epochs = step {stop_step} of {full_spe}/epoch")
    for tag, r in results.items():
        ge, de = r["greedy"]["enrichment"], r["digit_exp"]["enrichment"]
        gs = f"{ge['percentile']:.1f}" if ge else "n/a"
        ds = f"{de['percentile']:.1f}" if de else "n/a"
        print(f"{tag}: greedy {gs}%ile (beats_hard={r['greedy']['beats_hard']}) | "
              f"digit_exp {ds}%ile (beats_hard={r['digit_exp']['beats_hard']})")
    for name in ("greedy", "digit_exp"):
        ps = [r[name]["enrichment"]["percentile"] for r in results.values()
              if r[name]["enrichment"]]
        if ps:
            print(f"{name:10s} across seeds: {[round(p,1) for p in ps]} "
                  f"(spread {max(ps)-min(ps):.1f})")
        nb = sum(1 for r in results.values() if r[name]["beats_hard"])
        print(f"{name:10s} leak-free hard-negative test: {nb}/{len(results)} passed")
    print("\nCompare to try11's Fabry (11-molecule checking set): digit_exp enrichment")
    print("20.0 / 58.4 / 27.2 (spread 38.4), 1/3 passed. A tight spread here means the")
    print("inconclusiveness was the checking set; a wide spread means Fabry is genuinely")
    print("unstable and 89 compounds is simply not enough for this target.")
