"""try17 -- UNBLINDED fine-tuning, scored on the frozen equalized pool.

Copied from try14's step_stopping_eval.py and changed in exactly four places. Everything
else -- LoRA config, LR, epochs, evals per epoch, the scaffold-split validation, the
step-level stopping rule, the selection audit, and the score_both readout -- is
identical, so the reference drug's presence in training is the only variable.

  1. UNBLIND=1 injects the reference drug into the TRAINING pool with its own measured
     label and the same 5x SMILES augmentation prep gave every other compound.
  2. The reference is EXEMPT from the hard-negative scaffold holdout. Measured: on fabry
     and cel the reference shares a Bemis-Murcko scaffold with its own hard negatives, so
     without the exemption the holdout would delete the compound just injected and
     silently re-blind those two targets while labelling the run unblinded.
  3. The reference is pinned into the TRAIN split. A scaffold-based split would otherwise
     put it in validation on some seeds, re-blinding it for those seeds.
  4. The candidate set is the frozen 1,933-drug equalized pool (sha256-verified), not the
     500 cross-target decoys. Decoys are not scored at all. try15 measured a 31-point
     shift in a decoy percentile from a pool change alone; pool ranks are comparable
     across architectures and tries, decoy percentiles are not.

NO hard-negative result is produced. try14 established the selection is broken on 3 of 4
new targets -- it picks salts, tautomers and stereo-unannotated copies of the reference,
and on mastocytosis negatives genuinely more potent than the drug.

Reported per seed: the reference's rank in the pool under both readouts, the number of
pool compounds tied with it, and |predicted - real|. The label error matters because rank
is confounded with absolute potency: a CORRECT prediction for a weak binder such as
lumacaftor (1293 nM) should rank mid-pack, so a poor rank is not by itself a poor model.

DRY_RUN=1 stops after the data setup, before the model loads -- used to verify the
unblinding, the exemption and the split pinning on CPU.
"""
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

RDLogger.DisableLog("rdApp.*")

MODEL = "google/txgemma-9b-predict"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
# try14 results live in their own tree so this run cannot clobber try11's.
OUT_ROOT = os.path.join(DATA_DIR, "try17_unblinded", "stepstop")

TARGET = os.environ.get("TARGET")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
N_EPOCH = int(os.environ.get("N_EPOCH", "2"))
# try09 used 2e-4 and its optimum landed before the first measurement. Default is 1/3 of
# that. If the minimum STILL lands at the first eval step, 1/10 (2e-5) is the follow-up --
# readable directly off the saved curve.
LR = float(os.environ.get("LR", str(2e-4 / 3)))
EVALS_PER_EPOCH = int(os.environ.get("EVALS_PER_EPOCH", "7"))
# Validation forwards cost the same as training forwards, so the frequent step-level
# evaluations use ONE VARIANT PER VALIDATION COMPOUND rather than a subsample of lines.
#
# Why: augmentation is measured at 5.00x (fabry/gaucher/attr) and 4.92x (cf -- a few
# compounds got fewer than 5 variants where random SMILES generation produced duplicates,
# which also explains cf's 1,450-vs-1,427 count discrepancy). Subsampling 400 LINES would
# therefore cover only ~80 distinct molecules, and a curve measured on 80 molecules is
# nearly meaningless no matter how cleanly it is measured. One variant per compound instead
# gives the MAXIMUM distinct molecules per unit of compute:
#     cf 214 molecules | gaucher 181 | attr 72 | fabry 11
# All are under the old 400-line budget, so no subsampling is needed for any target and the
# step-level curve covers every validation compound.
#
# Augmentation exists to stop the model keying on an exact SMILES string during TRAINING;
# for measuring generalization one representation per molecule is sufficient. The full
# all-variants validation set is still measured at the selected and runner-up checkpoints.
VAL_EVAL_CAP = int(os.environ.get("VAL_EVAL_CAP", "400"))
N_DECOYS = int(os.environ.get("N_DECOYS", "500"))
GRAD_ACCUM = 8
VAL_FRACTION = 0.15

# --------------------------------------------------------------------- try17 additions
# UNBLIND=1 injects the reference drug's own measured label into the TRAINING pool. This
# is the whole point of this try: it measures the ceiling -- what the fine-tune achieves
# when the answer it is asked for is in its training data. UNBLIND=0 reproduces try14.
UNBLIND = os.environ.get("UNBLIND", "1") == "1"

# Scoring target: the frozen 1,933-drug equalized pool, sha256-verified, NOT the 500
# cross-target decoys. Ranks on this pool are comparable across architectures and across
# tries; decoy percentiles are not (try15 measured a 31-point shift from a pool change
# alone). Decoys are not scored here at all.
POOL_DIR = "/work/hdd/bisa/sshah21/results/equalized_pool"
POOL_JSON = os.path.join(POOL_DIR, "pool_v1.json")
POOL_SHA = os.path.join(POOL_DIR, "pool_v1.sha256")

N_SMILES_AUGMENT = 4  # matches prep_orphan_target_data.py, so the reference's injected
# training lines are built exactly the way every other compound's were
# try14 adds the new target names. mh and pku are deliberately absent: they are reported
# untrainable at prep time and never get a train_blind.jsonl, so including them here would
# only make build_decoys fail on a missing file.
#
# NOTE ON COMPARABILITY: the decoy field is built from every OTHER target's compounds, so
# adding four names changes the pool for every target. try14's percentiles are therefore
# comparable WITHIN try14 but not strictly against try11's numbers, which drew from a
# 4-target pool. The pool size actually used is printed and recorded per run.
ALL_TARGETS = ["gaucher", "cf", "fabry", "attr", "cel", "mastocytosis", "nsiad", "tyrosinemia"]

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


def augment_smiles(smiles, n=N_SMILES_AUGMENT):
    """[canonical, random_variant, ...] -- copied verbatim from
    prep_orphan_target_data.py so the reference's injected training lines are built the
    same way every other training compound's were. Using a different augmentation for
    the one compound under test would confound the experiment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    variants = {smiles}
    for _ in range(n * 3):
        if len(variants) > n:
            break
        variants.add(Chem.MolToSmiles(mol, doRandom=True))
    return list(variants)


def verify_pool():
    """Exits 2 on mismatch. Every arm must score the identical frozen pool, or the
    cross-architecture comparison this file feeds is meaningless."""
    with open(POOL_SHA) as fh:
        expected = fh.read().split()[0].strip()
    h = hashlib.sha256()
    with open(POOL_JSON, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        print(f"FATAL: pool sha256 mismatch\n  expected {expected}\n  actual {actual}",
              flush=True)
        sys.exit(2)
    pool = json.load(open(POOL_JSON))
    print(f"pool verified: {len(pool)} records, sha256 {actual}", flush=True)
    seen, out = set(), []
    for r in pool:
        c = canon(r["canonical_smiles"])
        if c is None or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def rank_in_pool(ref_score, all_scores):
    """Best-case rank (1 + strictly-better) and the number tied with the reference.
    Ties are reported because greedy collapses hard -- try16 measured up to 508 pool
    compounds sharing the reference's greedy value -- so breaking every tie in the
    reference's favour flatters greedy enormously."""
    if ref_score is None:
        return None, None
    vals = [v for v in all_scores if v is not None]
    if not vals:
        return None, None
    return sum(1 for v in vals if v < ref_score) + 1, sum(1 for v in vals if v == ref_score)


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
    # Top TWO checkpoints are kept, not just the best. The runner-up gets a full
    # all-variants validation pass at the end: if full-val disagrees with the step-level
    # val on which of the two is better, the step-level measurement is too noisy to select
    # on. try09's epoch-level steps differed by ~0.7 in val loss, which any reasonable
    # validation set resolves -- but the curve INSIDE epoch 1 may be far flatter, and that
    # is exactly where selection now happens.
    best = {"step": 0, "val": v0, "state": lora_state(model), "epoch": -1}
    runner = {"step": None, "val": float("inf"), "state": None, "epoch": None}
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
                        runner = {k: best[k] for k in ("step", "val", "state", "epoch")}
                        best = {"step": gstep, "val": vl, "state": lora_state(model),
                                "epoch": epoch}
                        flag = "  <- best"
                    elif vl < runner["val"]:
                        runner = {"step": gstep, "val": vl, "state": lora_state(model),
                                  "epoch": epoch}
                        flag = "  <- runner-up"
                    print(f"    step {gstep} (epoch {epoch}): train {tr:.4f}  "
                          f"val {vl:.4f}{flag}", flush=True)
        opt.step()
        opt.zero_grad()

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    model.eval()
    print(f"  SELECTED step {best['step']} (epoch {best['epoch']}), val {best['val']:.4f}, "
          f"from {len(curve)} measurements", flush=True)
    if runner["step"] is not None:
        print(f"  runner-up: step {runner['step']} (epoch {runner['epoch']}), "
              f"val {runner['val']:.4f}, gap {runner['val'] - best['val']:.4f}")
    return curve, best, runner


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


# build_decoys() and enrichment() were deleted with the decoy arm: the candidate set
# is now the frozen pool, and leaving an unused decoy builder in a snapshot invites a
# future reader to think decoys were scored here.


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

    # ------------------------------------------------------------------ UNBLINDING
    # train_blind.jsonl has the reference held out by prep. Injecting it here, with its
    # own measured label and the same 5x augmentation every other compound got, is what
    # "unblinded" means. Nothing else about the data changes, so the reference's
    # presence is the only variable between this run and try14's.
    ref_real = eval_set["reference"]["score"]
    if UNBLIND:
        if ref_smiles in groups:
            print(f"!! {TARGET}: reference was ALREADY in train_blind.jsonl -- blinding "
                  f"was broken before this try touched it", flush=True)
        ref_variants = augment_smiles(ref_smiles)
        groups[ref_smiles] = [{"prompt": prompt_for(template, v), "answer": str(ref_real)}
                              for v in ref_variants]
        print(f"UNBLINDED: injected reference {eval_set['reference']['name']!r} as "
              f"{len(ref_variants)} training lines with its real label {ref_real}",
              flush=True)
    else:
        print("BLINDED run (UNBLIND=0): reproducing try14's training data", flush=True)

    hard_scaffolds = {scaffold_of(h["smiles"]) for h in hard}
    hard_scaffolds.discard(None)
    scaffolds = {c: scaffold_of(c) for c in groups}
    removed = {c for c, s in scaffolds.items() if s in hard_scaffolds}
    # The reference is EXEMPT from the scaffold holdout when unblinding. It shares a
    # Bemis-Murcko scaffold with its own hard negatives on fabry and cel (measured), so
    # without this exemption the holdout would delete the compound we just injected and
    # silently re-blind exactly those two targets while reporting them as unblinded.
    if UNBLIND and ref_smiles in removed:
        removed.discard(ref_smiles)
        print(f"  reference shares a scaffold with its hard negatives; EXEMPTED from "
              f"the holdout so the unblinding is real. Hard negatives remain held out, "
              f"and this try reports no hard-negative result anyway.", flush=True)
    trainable = {c: v for c, v in groups.items() if c not in removed}
    print(f"=== {TARGET}: {len(groups)} compounds; reference in training? "
          f"{ref_smiles in trainable} (expect {UNBLIND})")
    if UNBLIND and ref_smiles not in trainable:
        print("FATAL: unblinding requested but the reference is not in the training "
              "pool. Refusing to report an 'unblinded' number that is not.", flush=True)
        sys.exit(6)
    print(f"hard negatives in training before holdout: "
          f"{sum(1 for h in hard if h['smiles'] in groups)}/{len(hard)}; after: "
          f"{sum(1 for h in hard if h['smiles'] in trainable)}/{len(hard)} (expect 0)")
    print(f"scaffold holdout removed {len(removed)} compounds "
          f"({100 * len(removed) / len(groups):.1f}%) -> {len(trainable)} trainable")

    pool_smis = verify_pool()
    if ref_smiles not in set(pool_smis):
        print(f"{TARGET}: reference {eval_set['reference']['name']!r} is NOT in pool_v1, "
              f"so it cannot be ranked on this pool. Nothing is substituted -- see "
              f"try16's README. Exiting without training: an unblinded adapter that "
              f"cannot be scored on the shared pool has nothing to report.", flush=True)
        os.makedirs(OUT_ROOT, exist_ok=True)
        with open(os.path.join(OUT_ROOT, f"{TARGET}_NOT_EVALUABLE.json"), "w") as f:
            json.dump({"target": TARGET, "evaluable": False,
                       "reason": "reference absent from pool_v1"}, f, indent=2)
        sys.exit(0)
    ref_pool_idx = pool_smis.index(ref_smiles)
    print(f"candidate set: the frozen equalized pool, {len(pool_smis)} approved drugs; "
          f"reference sits at index {ref_pool_idx}", flush=True)

    if os.environ.get("DRY_RUN") == "1":
        print("\nDRY_RUN=1: data setup verified, stopping before the model loads.")
        print(f"  unblinded          : {UNBLIND}")
        print(f"  reference in pool  : True (index {ref_pool_idx})")
        print(f"  trainable compounds: {len(trainable)}")
        print(f"  reference trainable: {ref_smiles in trainable}")
        print(f"  reference lines    : {len(trainable.get(ref_smiles, []))}")
        sys.exit(0)

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
        # The reference must TRAIN, not validate. A scaffold-based split would otherwise
        # put it in validation on some seeds, which re-blinds it for those seeds while
        # the run still calls itself unblinded.
        if UNBLIND and ref_smiles in vset:
            vset.discard(ref_smiles)
            val_c = [c for c in val_c if c != ref_smiles]
            print(f"  reference landed in the validation split for seed {seed}; moved "
                  f"to training so the unblinding holds for every seed", flush=True)
        train_c = [c for c in trainable if c not in vset]
        train_ex = [ex for c in train_c for ex in trainable[c]]
        val_ex = [ex for c in val_c for ex in trainable[c]]
        # ONE variant per validation compound for the step-level curve -- maximum distinct
        # molecules per unit of compute. Capped only as a safety net; no target reaches it.
        val_eval_pairs = [(c, trainable[c][0]) for c in val_c][:VAL_EVAL_CAP]
        val_eval_ex = [ex for _, ex in val_eval_pairs]
        n_distinct_eval = len({c for c, _ in val_eval_pairs})
        aug = len(val_ex) / max(1, len(val_c))
        print(f"\n=== {tag}: train {len(train_c)} compounds ({len(train_ex)} lines) | "
              f"val {len(val_c)} compounds ({len(val_ex)} lines, {aug:.2f}x augmented) | "
              f"{len(keys)} scaffolds ===", flush=True)
        print(f"  step-level curve measured on {len(val_eval_ex)} lines = "
              f"{n_distinct_eval} DISTINCT molecules "
              f"({100 * n_distinct_eval / max(1, len(val_c)):.0f}% of val compounds)",
              flush=True)
        if n_distinct_eval < 30:
            print(f"  WARNING: only {n_distinct_eval} distinct validation molecules. The "
                  f"curve will be noisy and its selected step is weak evidence.")

        base = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="cuda:0", dtype=torch.bfloat16)
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        model = get_peft_model(base, LORA_CONFIG, adapter_name=tag)

        curve, best, runner = train_step_level(
            model, tok, train_ex, val_eval_ex, N_EPOCH, LR, EVALS_PER_EPOCH)
        best_step, best_val, best_epoch = best["step"], best["val"], best["epoch"]
        device = next(model.parameters()).device

        # Full ALL-VARIANTS validation at the selected checkpoint (model currently holds
        # the best weights), and at the runner-up. If the full set disagrees with the
        # step-level curve about which of the two is better, the step-level measurement is
        # too noisy to select on -- and that has to be known, not assumed.
        full_val_best = val_loss(model, tok, val_ex, device)
        print(f"  full validation set ({len(val_ex)} lines) at SELECTED step "
              f"{best_step}: {full_val_best:.4f}")

        selection_check = {"agrees": None, "full_val_best": full_val_best,
                           "full_val_runner": None, "runner_step": runner["step"]}
        if runner["state"] is not None:
            model.load_state_dict(runner["state"], strict=False)
            full_val_runner = val_loss(model, tok, val_ex, device)
            agrees = full_val_best <= full_val_runner
            selection_check.update({"agrees": agrees, "full_val_runner": full_val_runner})
            print(f"  full validation set at RUNNER-UP step {runner['step']}: "
                  f"{full_val_runner:.4f}")
            if agrees:
                print(f"  SELECTION CHECK: full-val agrees with the step-level curve "
                      f"(selected {full_val_best:.4f} <= runner-up {full_val_runner:.4f})")
            else:
                print(f"  SELECTION CHECK FAILED: full-val RANKS THE RUNNER-UP BETTER "
                      f"({full_val_runner:.4f} < {full_val_best:.4f}). The step-level "
                      f"curve mis-selected -- {n_distinct_eval} distinct validation "
                      f"molecules is too few to resolve this curve. Treat this seed's "
                      f"selected step as unreliable and widen the validation set.")
            # Restore the selected weights before scoring, whatever the check said --
            # selection policy stays 'lowest step-level val', and the check is diagnostic.
            model.load_state_dict(best["state"], strict=False)
            model.eval()

        # try09 lost its models by never saving. Save the selected weights.
        adapter_path = os.path.join(out_dir, f"{tag}_adapter")
        model.save_pretrained(adapter_path, selected_adapters=[tag])
        print(f"  saved adapter -> {adapter_path}")
        full_val = full_val_best

        print(f"=== Scoring {tag}: both readouts, one forward pass each ===", flush=True)
        ref_g, ref_d = score_both(model, tok, prompt_for(template, ref_smiles), dmap)
        print(f"  reference: greedy={ref_g} "
              f"digit_exp={round(ref_d, 1) if ref_d is not None else None} "
              f"(real {eval_set['reference']['score']})")

        # Hard negatives are NOT scored and beats_hard is NOT computed. try14 showed
        # the selection is broken -- it picks salts, tautomers and stereo-unannotated
        # copies of the reference, and on mastocytosis negatives that are genuinely more
        # potent than the drug. Reporting that flag beside an honest pool rank would
        # mislead. The pool rank is this try's result.
        dg, dd = [], []
        for i, smi in enumerate(pool_smis):
            g, d = score_both(model, tok, prompt_for(template, smi), dmap)
            dg.append(g)
            dd.append(d)
            if (i + 1) % 400 == 0:
                print(f"    {i + 1}/{len(pool_smis)} pool compounds", flush=True)

        res = {"seed": seed, "lr": LR, "n_epoch": N_EPOCH,
               "n_train_compounds": len(train_c), "n_val_compounds": len(val_c),
               "n_train_lines": len(train_ex), "validation_curve": curve,
               "selected_step": best_step, "selected_epoch": best_epoch,
               "selected_val_loss": best_val, "full_val_loss_at_selected": full_val,
               "runner_up_step": runner["step"], "runner_up_val_loss": runner["val"],
               "selection_check": selection_check,
               "n_val_lines": len(val_ex), "augmentation_factor": aug,
               "n_distinct_val_molecules_in_curve": n_distinct_eval,
               "adapter_path": adapter_path,
               "unblinded": UNBLIND,
               "reference_in_training": ref_smiles in set(train_c),
               "n_pool": len(pool_smis),
               "reference_real": ref_real}
        for name, refv, dvals in (("greedy", ref_g, dg), ("digit_exp", ref_d, dd)):
            rank, tied = rank_in_pool(dvals[ref_pool_idx], dvals)
            # Label recovery is the honest unblinded question: given the answer in
            # training, does the model reproduce it? Rank alone is confounded -- a
            # correct prediction for a weak binder like lumacaftor (1293 nM) SHOULD
            # rank mid-pack, so a poor rank there is not necessarily a poor model.
            err = None if refv is None else abs(refv - ref_real)
            res[name] = {
                "reference": refv,
                "reference_from_pool_scoring": dvals[ref_pool_idx],
                "abs_label_error": err,
                "rank": rank, "n_pool": len(pool_smis),
                "percentile": round(100.0 * rank / len(pool_smis), 3) if rank else None,
                "n_tied_with_reference": tied,
                "n_distinct_scores": len({v for v in dvals if v is not None}),
                "n_unscored": sum(1 for v in dvals if v is None),
            }
            if rank:
                print(f"  POOL RANK [{name}]: {rank}/{len(pool_smis)} = "
                      f"{100.0 * rank / len(pool_smis):.2f} percentile | tied {tied} | "
                      f"distinct {res[name]['n_distinct_scores']} | "
                      f"predicted {refv} vs real {ref_real} (err {err})", flush=True)
        results[tag] = res

        with open(os.path.join(out_dir, f"{tag}_pool_scores.json"), "w") as f:
            json.dump({"pool_smiles": pool_smis, "reference_pool_index": ref_pool_idx,
                       "greedy": dg, "digit_exp": dd}, f)
        del model, base
        torch.cuda.empty_cache()
        with open(os.path.join(out_dir, "step_stopping_summary.json"), "w") as f:
            json.dump({"target": TARGET, "seeds": SEEDS, "lr": LR, "n_epoch": N_EPOCH,
                       "evals_per_epoch": EVALS_PER_EPOCH, "results": results}, f, indent=2)

    print(f"\n=== FINAL SUMMARY: {TARGET} (step-level stopping, lr={LR:.2e}, "
          f"{len(SEEDS)} seeds) ===")
    print(f"candidate set: equalized pool, {results[next(iter(results))]['n_pool']} "
          f"approved drugs. UNBLIND={UNBLIND}.")
    header = (f"{'run':16s} {'sel.step':>9s} {'val':>7s} {'g.rank':>8s} {'g%ile':>7s} "
              f"{'d.rank':>8s} {'d%ile':>7s} {'d.pred':>7s} {'real':>6s} {'err':>6s}")
    print(header)
    for tag, r in results.items():
        g, d = r["greedy"], r["digit_exp"]
        dp = f"{d['reference']:.0f}" if d["reference"] is not None else "n/a"
        de = f"{d['abs_label_error']:.0f}" if d["abs_label_error"] is not None else "n/a"
        print(f"{tag:16s} {r['selected_step']:>9} {r['selected_val_loss']:>7.3f} "
              f"{str(g['rank']):>8s} {g['percentile'] or 0:>7.2f} "
              f"{str(d['rank']):>8s} {d['percentile'] or 0:>7.2f} "
              f"{dp:>7s} {r['reference_real']:>6} {de:>6s}")

    mis = [t for t, r in results.items()
           if r["selection_check"].get("agrees") is False]
    if mis:
        print(f"\nSELECTION CHECK FAILED for {mis} -- full all-variants validation ranked "
              f"the runner-up checkpoint better than the selected one. The step-level "
              f"curve was measured on {results[mis[0]]['n_distinct_val_molecules_in_curve']} "
              f"distinct molecules, which is too few to resolve it. Widen the validation "
              f"set before trusting those selected steps.")
    else:
        print("\nselection check: full all-variants validation agreed with the step-level "
              "curve on the top-two ordering for every seed.")
    print(f"distinct validation molecules in the curve: "
          f"{[r['n_distinct_val_molecules_in_curve'] for r in results.values()]} "
          f"(augmentation {results[next(iter(results))]['augmentation_factor']:.2f}x)")

    steps = [r["selected_step"] for r in results.values()]
    print(f"selected steps: {steps}")
    if steps and max(steps) == 0:
        print("  WARNING: every seed selected step 0, the PRE-TRAINING checkpoint. Training")
        print("  never improved held-out loss at all, so these adapters are effectively the")
        print("  base model. Lower LR further (2e-5) before concluding anything about")
        print("  fine-tuning -- or the task is not learnable from this data at this scale.")
    elif steps and min(steps) <= 1:
        print("  NOTE: a seed selected step <=1, so the optimum is still at the very start.")
        print("  Re-run at lr=2e-5 (1/10 of try09's) before trusting these adapters.")
    for name in ("greedy", "digit_exp"):
        ps = [r[name]["percentile"] for r in results.values() if r[name]["percentile"]]
        if ps:
            print(f"{name:10s} pool percentile across seeds: {[round(p, 2) for p in ps]} "
                  f"(spread {max(ps) - min(ps):.2f})")
        es = [r[name]["abs_label_error"] for r in results.values()
              if r[name]["abs_label_error"] is not None]
        if es:
            print(f"{name:10s} |predicted - real| across seeds: "
                  f"{[round(e) for e in es]} (mean {sum(es) / len(es):.0f})")
        ties = [r[name]["n_tied_with_reference"] for r in results.values()]
        print(f"{name:10s} pool compounds tied with the reference: {ties}")
    print("\nNo hard-negative result is reported: try14 showed the selection is broken "
          "on 3 of 4 new targets. The pool rank and the label error are this try's "
          "numbers.")
    print("CAUTION on rank: it is confounded with absolute potency. A correct "
          "prediction for a weak binder (ambroxol 10979 nM, lumacaftor 1293 nM, "
          "tafamidis 839 nM) SHOULD rank mid-pack or low, so for those targets read the "
          "label error, not the rank.")
