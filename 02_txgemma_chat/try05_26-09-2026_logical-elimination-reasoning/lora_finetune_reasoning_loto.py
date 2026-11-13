"""
Leave-one-target-out fine-tune + eval for txgemma-27b-chat reasoning.

For each of the 4 targets {fabry, gaucher, cf, attr}: train a LoRA adapter on
augmented reasoning-chain examples from the OTHER THREE targets only, then evaluate
COLD on the held-out target's original (unaugmented) example -- its sequence,
mechanism description, and reference drug never appear anywhere in that fold's
training data, not even as a distractor in another target's candidate list. This
tests cross-mechanism transfer, not within-target interpolation (holding out a
compound but keeping its target's other data in training -- what we did for the
numeric predict task -- is a much easier, less meaningful test; see conversation).

Training-scenario augmentation: with only 3 base examples per fold, fine-tuning would
have almost nothing to generalize from and a high chance of memorizing surface
patterns. For each training target we generate N_AUGMENT_PER_TARGET variants by
combining: a mechanism-description paraphrase (same facts, different wording -- same
anti-memorization principle as the SMILES augmentation used for the numeric task), a
random subset of that target's own real hard negatives, a random subset of the OTHER
training targets' real reference drugs as cross-mechanism distractors (never the held-
out target's), a random subset of easy negatives, and a fresh random blind-label
ordering.

v2 (26-09-2026): the completion itself is no longer a single fixed paragraph per
target (author_reasoning_examples.py's docstring has the full root-cause writeup --
the model was memorizing that one paragraph verbatim and reciting it, even when it
didn't fit the held-out prompt). The completion now states an explicit necessary
condition for the mechanism, then gives every individual candidate a
satisfies/fails/partial verdict checked against that condition, built compositionally
from (scaffold class x mechanism condition) rather than from one fixed narrative --
so which exact verdicts appear varies by which candidates got sampled into a given
augmented instance, and there's no single block left to memorize wholesale. Cold-eval
scoring now reports a second metric alongside top-choice correctness:
logic_consistency, the fraction of the model's own per-candidate verdicts that agree
with ground truth -- this is the more direct test of whether it can apply the
compositional rule to an unseen target, independent of whether the final ranking
happened to come out right.

Honest caveat, stated here so it travels with the results: 4 verified mechanism-class
examples is a thin base to draw 3 from per fold. A pass here is suggestive evidence of
transfer, not proof of general orphan-disease reasoning ability.
"""
import json
import os
import random
import sys
import zlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from author_reasoning_examples import (
    REFERENCE_NAMES, MECHANISM_DESCRIPTIONS, MECHANISM_PARAPHRASES,
    FASTA_PATHS, EASY_NEGATIVES, read_fasta, build_prompt, build_completion,
)

MODEL = "google/txgemma-27b-chat"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
REASONING_DIR = "/work/hdd/bisa/sshah21/results/reasoning_chains"
OUT_DIR = "/work/hdd/bisa/sshah21/results/reasoning_loto"
N_AUGMENT_PER_TARGET = 12
TARGETS = list(REFERENCE_NAMES.keys())

LORA_CONFIG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def load_eval_sets():
    return {name: json.load(open(os.path.join(DATA_DIR, name, "eval_set.json"))) for name in TARGETS}


def make_augmented_instance(name, eval_sets, reference_smiles, allowed_cross_targets, aug_idx):
    """One augmented (prompt, completion_json_str) pair for training target `name`.
    allowed_cross_targets excludes the held-out target for this fold entirely."""
    rng = random.Random(zlib.crc32(f"{name}_{aug_idx}".encode()))

    es = eval_sets[name]
    seq = read_fasta(FASTA_PATHS[name])
    mechanism = rng.choice([MECHANISM_DESCRIPTIONS[name]] + MECHANISM_PARAPHRASES[name])

    hard_negs = es["hard_negatives"]
    chosen_hard = rng.sample(hard_negs, rng.randint(1, len(hard_negs)))

    cross_pool = [t for t in allowed_cross_targets if t != name]
    chosen_cross = rng.sample(cross_pool, rng.randint(1, len(cross_pool))) if cross_pool else []

    easy_names = list(EASY_NEGATIVES.keys())
    chosen_easy = rng.sample(easy_names, rng.randint(1, len(easy_names)))

    candidates = {REFERENCE_NAMES[name]: es["reference"]["smiles"]}
    for i, hn in enumerate(chosen_hard):
        candidates[f"{name}_analog_{i + 1}"] = hn["smiles"]
    for t in chosen_cross:
        candidates[REFERENCE_NAMES[t]] = reference_smiles[t]
    for e in chosen_easy:
        candidates[e] = EASY_NEGATIVES[e]

    names_list = list(candidates.keys())
    shuffled = names_list[:]
    rng.shuffle(shuffled)
    labels = [f"Compound {chr(ord('A') + i)}" for i in range(len(shuffled))]
    label_to_name = dict(zip(labels, shuffled))
    name_to_label = {v: k for k, v in label_to_name.items()}

    prompt = build_prompt(seq, mechanism, label_to_name, candidates)
    completion = build_completion(name, names_list, name_to_label, rng)
    return prompt, json.dumps(completion)


def make_batch(tok, examples, device):
    """One example = (prompt, completion_json_str). Builds chat-templated input_ids
    with labels masked (-100) over everything except the completion."""
    input_ids_list, labels_list = [], []
    for prompt, completion in examples:
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]
        completion_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
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


def train_adapter(model, tok, examples, n_epoch=3, lr=1e-4, grad_accum=4):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = model.device  # NOT next(model.parameters()).device -- unreliable for a
    # device_map="auto"-sharded model; model.device is the pattern already proven to
    # work for this exact model in chat_txgemma.py / blind_chaperone_rank_txgemma.py.
    for epoch in range(n_epoch):
        random.shuffle(examples)
        total_loss, step = 0.0, 0
        opt.zero_grad()
        for ex in examples:
            input_ids, attn, labels = make_batch(tok, [ex], device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss / grad_accum).backward()
            total_loss += out.loss.item()
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        print(f"  epoch {epoch}: avg loss {total_loss / step:.4f}")
    model.eval()


@torch.no_grad()
def generate_completion(model, tok, prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_json(text):
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    eval_sets = load_eval_sets()
    reference_smiles = {name: es["reference"]["smiles"] for name, es in eval_sets.items()}

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    fold_results = {}
    for held_out in TARGETS:
        train_targets = [t for t in TARGETS if t != held_out]
        print(f"\n=== FOLD: holding out '{held_out}', training on {train_targets} ===")

        examples = []
        for name in train_targets:
            for aug_idx in range(N_AUGMENT_PER_TARGET):
                examples.append(make_augmented_instance(name, eval_sets, reference_smiles,
                                                          allowed_cross_targets=train_targets,
                                                          aug_idx=aug_idx))
        print(f"  {len(examples)} augmented training instances "
              f"({N_AUGMENT_PER_TARGET} per target x {len(train_targets)} targets)")

        base_model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", dtype=torch.bfloat16)
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        model = get_peft_model(base_model, LORA_CONFIG, adapter_name=held_out)

        print(f"=== Training (holding out {held_out}) ===")
        train_adapter(model, tok, examples)
        fold_dir = os.path.join(OUT_DIR, f"holdout_{held_out}")
        os.makedirs(fold_dir, exist_ok=True)
        model.save_pretrained(os.path.join(fold_dir, "adapter"), selected_adapters=[held_out])

        print(f"=== Cold eval on held-out target: {held_out} ===")
        held_out_example = json.load(open(os.path.join(REASONING_DIR, f"{held_out}.json")))
        reply = generate_completion(model, tok, held_out_example["prompt"])
        parsed = extract_json(reply)
        top_label = parsed.get("top_choice") if parsed else None
        top_name = held_out_example["label_to_name"].get(top_label) if top_label else None
        correct = top_name == REFERENCE_NAMES[held_out]

        # Logic-consistency check: independent of whether the final ranking was correct,
        # did the model's own per-candidate satisfies/fails/partial verdicts agree with
        # the ground-truth verdicts (built from the same scaffold-class-vs-condition
        # table used in training)? This is the thing this redesign is actually meant to
        # test -- can it apply the compositional rule per candidate, not just recite a
        # memorized final answer.
        gt_verdicts = held_out_example["completion"]["candidate_verdicts"]
        model_verdicts = parsed.get("candidate_verdicts", {}) if parsed else {}
        n_matched = sum(
            1 for label, gt in gt_verdicts.items()
            if label in model_verdicts and model_verdicts[label].get("satisfies_condition") == gt["satisfies_condition"]
        )
        logic_consistency = n_matched / len(gt_verdicts)

        print(f"  expected: {REFERENCE_NAMES[held_out]} ({[l for l, n in held_out_example['label_to_name'].items() if n == REFERENCE_NAMES[held_out]][0]})")
        print(f"  model top_choice: {top_label} -> {top_name} | correct: {correct}")
        print(f"  logic consistency: {n_matched}/{len(gt_verdicts)} candidate verdicts matched ground truth ({logic_consistency:.2f})")
        print(f"  raw reply: {reply}")

        fold_results[held_out] = {
            "expected": REFERENCE_NAMES[held_out], "top_choice_label": top_label,
            "top_choice_name": top_name, "correct": correct,
            "logic_consistency": logic_consistency, "raw_reply": reply,
            "n_training_instances": len(examples),
        }
        with open(os.path.join(fold_dir, "result.json"), "w") as f:
            json.dump(fold_results[held_out], f, indent=2)

        del model, base_model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, "loto_summary.json"), "w") as f:
        json.dump(fold_results, f, indent=2)

    print("\n=== LOTO SUMMARY ===")
    for held_out, r in fold_results.items():
        print(f"held out {held_out}: expected={r['expected']} got={r['top_choice_name']} "
              f"correct={r['correct']} logic_consistency={r['logic_consistency']:.2f}")
    n_correct = sum(1 for r in fold_results.values() if r["correct"])
    avg_logic = sum(r["logic_consistency"] for r in fold_results.values()) / len(fold_results)
    print(f"\n{n_correct}/{len(fold_results)} folds correct. "
          f"Average logic consistency: {avg_logic:.2f} (fraction of per-candidate "
          f"satisfies/fails/partial verdicts matching ground truth, independent of "
          f"whether the final top choice was correct). "
          f"(Caveat: each fold trains on just 3 examples' worth of augmented scenarios -- "
          f"a thin base. Treat this as suggestive, not conclusive.)")
