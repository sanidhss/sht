"""
Diagnostic (not a new fine-tune): does the holdout_fabry adapter -- trained on
gaucher/cf/attr, saved by try06's job 22544773 -- correctly derive an IN-TRAINING
target's own necessary_condition when prompted with that target's own prompt, or does
it show the same blending/retrieval failure seen cold on the held-out target?

If it grounds cleanly in-distribution (gaucher/cf/attr, all seen in this adapter's
training) but not cold (fabry, the held-out target, where try06 showed it blending in
gaucher's "aminocyclohexanol" language and confabulating a "di-metal-binding hexose"
detail present in no authored text), that confirms the failure is specifically a
distribution-shift / retrieval-over-derivation problem, not a general instability in
the model's ability to read a mechanism description and state a condition from it.

Uses the already-saved adapter and already-saved reasoning_chains/*.json prompts --
no retraining.
"""
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "google/txgemma-27b-chat"
ADAPTER_DIR = "/work/hdd/bisa/sshah21/results/reasoning_loto/holdout_fabry/adapter/fabry"
REASONING_DIR = "/work/hdd/bisa/sshah21/results/reasoning_chains"

# This adapter was trained holding fabry out (on gaucher/cf/attr). Test it on one
# in-training target (gaucher) and the held-out one (fabry) for direct comparison.
IN_TRAINING_TARGET = "gaucher"
HELD_OUT_TARGET = "fabry"


def extract_json(text):
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@torch.no_grad()
def generate_completion(model, tok, prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    for target, label in [(IN_TRAINING_TARGET, "IN-TRAINING"), (HELD_OUT_TARGET, "HELD-OUT (cold)")]:
        example = json.load(open(os.path.join(REASONING_DIR, f"{target}.json")))
        print(f"\n=== {label}: {target} ===")
        print(f"authored ground-truth necessary_condition: {example['completion']['necessary_condition']!r}")
        reply = generate_completion(model, tok, example["prompt"])
        parsed = extract_json(reply)
        stated = parsed.get("necessary_condition") if parsed else None
        print(f"model's stated necessary_condition: {stated!r}")
        print(f"raw reply: {reply}")
