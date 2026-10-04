"""
Gaucher downsample test -- is the Fabry/TTR failure caused by training-data volume?

Background. Across all four targets, hard-negative pass/fail is monotonic in the
number of real training compounds:

    CF      1,427  PASS
    Gaucher 1,208  PASS
    TTR       493  FAIL
    Fabry      89  FAIL

That is suggestive but confounded: a better-studied target has both more data and
possibly cleaner data, and the four targets differ in mechanism. n=4.

This script breaks the confound. It takes Gaucher -- a target that PASSES at full
size -- and retrains it on a random 493-compound subset, matching TTR's compound
count exactly. Same target, same mechanism, same pipeline, same held-out reference
and same hard negatives. Training volume is the only thing that changes.

    If downsampled Gaucher collapses the way TTR does, volume is the bottleneck and
    Fabry + TTR become ONE data-sufficiency finding rather than two mechanism-specific
    failures.

    If downsampled Gaucher still passes, volume is NOT sufficient explanation, and the
    TTR/Fabry failures need a mechanism- or data-quality-based account instead. That
    outcome would redirect the whole next phase, so it is just as informative.

Three seeds, because a single 493-draw could be unlucky (or lucky) in which compounds
it happens to retain. A 1-of-3 or 2-of-3 split is itself a meaningful signal that 493
sits near the threshold rather than clearly below it.

Design notes / why it is built this way:

  - No re-prep, no network. Gaucher's train_blind.jsonl is already prepped and
    augmented (1,208 compounds x 5 SMILES variants = 6,040 lines). Re-running
    prep_orphan_target_data.py would re-hit ChEMBL/BindingDB/PubChem and risk the
    underlying data shifting under the comparison, which would defeat the point.

  - Downsampling happens at the DISTINCT COMPOUND level, not the line level. Sampling
    6,040 lines down to 2,465 directly would cut augmentation depth at the same time
    as compound diversity, confounding the two. Augmented variants are regrouped by
    RDKit canonical SMILES (verified: recovers exactly 1,208 groups of exactly 5, one
    consistent label each), then whole compounds are sampled, keeping all 5 of their
    variants. The result is 493 x 5 = 2,465 lines -- structurally identical to TTR's
    493 -> 2,463.

  - The eval set is reused byte-for-byte from the full-size Gaucher run. The reference
    (ambroxol) and the 3 real structural analogs are unchanged, so the numbers here are
    directly comparable to try05's full-size baseline.

  - Writes to gaucher_downsample/, NOT gaucher/. try05's adapter and eval set are left
    untouched; this run must not overwrite the passing baseline it is compared against.

KNOWN CONFOUND, stated here so a future reader does not over-read a collapse: 493
compounds at 5x augmentation means ~2,465 training lines and therefore ~2.4x fewer
gradient steps per epoch than full-size Gaucher. So a collapse here is consistent with
either "too few distinct compounds" or "too little training compute," which this run
cannot separate. It matches TTR's configuration exactly, which is what makes it the
right test of the TTR question specifically. Separating diversity from compute needs a
matched-compute arm (493 compounds at ~12x augmentation to restore ~6,040 lines); that
is the follow-up if this run collapses, not part of this run.
"""
import json
import os
import random
import re
from collections import defaultdict

import torch
from rdkit import Chem, RDLogger
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

RDLogger.DisableLog("rdApp.*")

MODEL = "google/txgemma-9b-predict"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
SOURCE = os.path.join(DATA_DIR, "gaucher")
OUT_DIR = os.path.join(DATA_DIR, "gaucher_downsample")

# 493 to match TTR's real compound count exactly -- this is the comparison that matters.
N_KEEP = 493
SEEDS = [0, 1, 2]

# try05's full-size Gaucher result, for the printed side-by-side. These are PREDICTED
# scores (ambroxol real label 685, analog real labels 791/679/839). The reference beat
# all three predicted analog scores; that is the behavior this run checks for the loss
# of.
#
# Worth knowing when reading any Gaucher result: analog #2's real label is 679, which
# is slightly MORE potent than ambroxol's real 685. So "reference beats all hard
# negatives" is not strictly the ground truth for this target -- the full-size model
# predicting ambroxol best is, on that one pair, ranking against the real labels. It
# does not affect this run's comparison (identical eval set on both sides, and the
# question here is whether discrimination survives downsampling, not whether the
# ranking is correct), but it does mean Gaucher's "PASS" is a slightly softer claim
# than it looks.
BASELINE = {"n_compounds": 1208, "reference_predicted": 691, "reference_real": 685,
            "hard_predicted": [791, 791, 839], "hard_real": [791, 679, 839]}

LORA_CONFIG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def group_by_compound(path):
    """Regroup augmented training lines back into one entry per distinct compound.

    The jsonl carries only {prompt, answer} -- no compound id -- so identity is
    recovered by pulling the SMILES out of the prompt and canonicalizing it. Random
    RDKit traversals of the same molecule canonicalize to the same string, which is
    what collapses the 5 augmented variants back together.
    """
    groups = defaultdict(list)
    unparseable = 0
    for line in open(path):
        d = json.loads(line)
        m = re.search(r"Drug SMILES: (.+)", d["prompt"])
        if not m:
            unparseable += 1
            continue
        mol = Chem.MolFromSmiles(m.group(1).strip())
        if mol is None:
            unparseable += 1
            continue
        groups[Chem.MolToSmiles(mol)].append(d)
    if unparseable:
        print(f"[warn] {unparseable} lines had no parseable SMILES and were dropped")
    return groups


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


def train_adapter(model, tok, examples, n_epoch=3, lr=2e-4, micro_bs=1, grad_accum=8):
    """Identical hyperparameters to try06's train_adapter -- deliberately unchanged, so
    any difference in outcome is attributable to the data, not the optimizer."""
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device
    losses = []
    for epoch in range(n_epoch):
        random.shuffle(examples)
        total_loss, step = 0.0, 0
        opt.zero_grad()
        for i in range(0, len(examples), micro_bs):
            batch = examples[i:i + micro_bs]
            input_ids, attn, labels = make_batch(tok, batch, device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            total_loss += out.loss.item()
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        avg = total_loss / step
        losses.append(avg)
        print(f"  epoch {epoch}: avg loss {avg:.4f}", flush=True)
    model.eval()
    return losses


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


def validate(model, tok, eval_set, tag):
    """Same validation as try06, plus an explicit collapse check.

    'Collapse' is the specific TTR/Fabry failure signature: one identical score emitted
    for the reference and every hard negative, i.e. the model has stopped discriminating
    rather than merely ranking wrongly. It is tracked separately from beats_hard because
    the two are different failures -- ranking the reference second is a weaker problem
    than being unable to tell four molecules apart at all.
    """
    ref = eval_set["reference"]
    ref_score = score(model, tok, ref["prompt"])
    print(f"[{tag}] reference '{ref['name']}' predicted: {ref_score!r} "
          f"(real label {ref['score']}, held out of training)")

    easy_scores = {}
    print(f"[{tag}] easy negatives:")
    for c in eval_set["easy_negatives"]:
        s = score(model, tok, c["prompt"])
        easy_scores[c["name"]] = s
        print(f"    {c['name']}: {s!r}")

    hard_scores = {}
    print(f"[{tag}] hard negatives (real structural analogs -- the meaningful test):")
    for c in eval_set["hard_negatives"]:
        label = c["name"] or c["smiles"][:20]
        s = score(model, tok, c["prompt"])
        hard_scores[label] = s
        print(f"    {label} (real label {c['score']}): {s!r}")

    ref_val = to_int(ref_score)
    beat_easy = ref_val is not None and all(
        to_int(v) is None or ref_val < to_int(v) for v in easy_scores.values())
    beat_hard = ref_val is not None and all(
        to_int(v) is None or ref_val < to_int(v) for v in hard_scores.values())

    hard_vals = [to_int(v) for v in hard_scores.values()]
    collapsed = (ref_val is not None and all(v == ref_val for v in hard_vals))

    print(f"[{tag}] beats_easy={beat_easy}  beats_hard={beat_hard}  collapsed={collapsed}")
    if collapsed:
        print(f"[{tag}] -> COLLAPSE: same score for reference and all analogs "
              f"(the TTR/Fabry signature)")
    elif not beat_hard:
        print(f"[{tag}] -> FAIL: discriminates, but does not rank the reference first")
    else:
        print(f"[{tag}] -> PASS: reference beats all real structural analogs")

    return {"reference_score": ref_score, "easy_scores": easy_scores,
            "hard_scores": hard_scores, "beats_easy": beat_easy,
            "beats_hard": beat_hard, "collapsed": collapsed}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    groups = group_by_compound(os.path.join(SOURCE, "train_blind.jsonl"))
    print(f"source: {len(groups)} distinct compounds "
          f"({sum(len(v) for v in groups.values())} augmented lines)")
    if len(groups) < N_KEEP:
        raise SystemExit(f"source has only {len(groups)} compounds, cannot sample {N_KEEP}")

    eval_set = json.load(open(os.path.join(SOURCE, "eval_set.json")))
    print(f"reusing full-size Gaucher eval set unchanged: reference="
          f"{eval_set['reference']['name']}, {len(eval_set['hard_negatives'])} hard negatives")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    all_keys = sorted(groups)
    results = {}

    for seed in SEEDS:
        tag = f"gaucher_n{N_KEEP}_seed{seed}"
        rng = random.Random(seed)
        kept = rng.sample(all_keys, N_KEEP)
        examples = [ex for k in kept for ex in groups[k]]
        print(f"\n=== {tag}: {len(kept)} compounds -> {len(examples)} training lines ===",
              flush=True)

        # Record exactly which compounds this seed retained, so a surprising result can
        # be inspected (or the split reproduced) without re-deriving it.
        with open(os.path.join(OUT_DIR, f"{tag}_compounds.json"), "w") as f:
            json.dump({"seed": seed, "n_compounds": len(kept), "n_lines": len(examples),
                       "compounds": kept}, f, indent=2)

        # Fresh base-model load per seed. Same reasoning as try06: reusing one base
        # across adapters previously produced an unexplained backwards result that was
        # never root-caused, and this run's whole value is being trusted, so pay the
        # extra load time rather than reintroduce that doubt.
        print(f"=== Loading base model for {tag} ===", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="cuda:0", dtype=torch.bfloat16)
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        model = get_peft_model(base_model, LORA_CONFIG, adapter_name=tag)

        print(f"=== Training {tag} ===", flush=True)
        losses = train_adapter(model, tok, examples)
        model.save_pretrained(os.path.join(OUT_DIR, f"{tag}_adapter"), selected_adapters=[tag])

        print(f"=== Validating {tag} ===", flush=True)
        r = validate(model, tok, eval_set, tag)
        r.update({"seed": seed, "n_compounds": len(kept), "n_lines": len(examples),
                  "epoch_losses": losses})
        results[tag] = r

        del model, base_model
        torch.cuda.empty_cache()

        # Written after every seed so a timeout or OOM on a later seed still leaves the
        # completed ones on disk -- try05 lost a whole run's output by only writing at
        # the end.
        with open(os.path.join(OUT_DIR, "downsample_summary.json"), "w") as f:
            json.dump({"n_keep": N_KEEP, "seeds": SEEDS, "baseline_full_size": BASELINE,
                       "results": results}, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(f"full-size Gaucher baseline (try05, n={BASELINE['n_compounds']}): "
          f"reference predicted={BASELINE['reference_predicted']} (real "
          f"{BASELINE['reference_real']}), analogs predicted="
          f"{BASELINE['hard_predicted']} (real {BASELINE['hard_real']}) -> PASS")
    for tag, r in results.items():
        print(f"{tag}: reference={r['reference_score']!r} "
              f"analogs={list(r['hard_scores'].values())} "
              f"beats_hard={r['beats_hard']} collapsed={r['collapsed']}")

    n_collapsed = sum(1 for r in results.values() if r["collapsed"])
    n_failed = sum(1 for r in results.values() if not r["beats_hard"])
    print(f"\n{n_collapsed}/{len(results)} seeds collapsed; "
          f"{n_failed}/{len(results)} failed the hard-negative test")
    print("Interpretation guide (full reasoning in this try's README):")
    print("  3/3 collapsed      -> volume is the bottleneck; Fabry+TTR are one finding")
    print("  0/3 collapsed      -> volume is NOT sufficient; the monotonic n_training")
    print("                        pattern across the 4 targets is confounded, and the")
    print("                        failures need a mechanism/data-quality account")
    print("  1-2/3 collapsed    -> 493 sits near the threshold, not clearly below it")
