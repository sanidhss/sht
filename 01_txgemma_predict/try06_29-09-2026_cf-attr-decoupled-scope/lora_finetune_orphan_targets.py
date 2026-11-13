"""
Generalized LoRA fine-tune + validation for orphan-disease targets.

For each target with data prepared by prep_orphan_target_data.py: fine-tune a
target-specific LoRA adapter on its ChEMBL training set (reference drug held
out entirely), then check whether the model ranks the reference drug above
BOTH:
  - the easy negatives (unrelated drug classes -- a weak test, passable by
    coarse chemotype recognition alone), AND
  - the hard negatives (real structurally-similar analogs from the same
    dataset that are empirically much weaker binders -- the test that
    actually indicates the model learned the right structure-activity
    signal, not just "this doesn't look like a drug for this target").

Only a target that passes the hard-negative check should be trusted to rank a
full drug library (e.g. all of DrugBank) for real candidate screening -- the
whole point of this script is to be that gate, run once per new orphan target
before committing orthogonal wet-lab validation effort to its output.

This does NOT auto-run prep_orphan_target_data.py and is NOT auto-submitted to
SLURM. Run prep first (CPU only), inspect coverage_report.json, then submit
this via lora_finetune_orphan_targets.sb only for targets marked "ready".
"""
import json
import os
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "google/txgemma-9b-predict"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"

LORA_CONFIG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def targets_to_run():
    """Returns {name: status} for targets worth attempting -- status is threaded
    through to the final verdict so a thin-data result is never worded the same
    as a well-supported one.

    Scoped by the TARGETS_TO_RUN env var (comma-separated, e.g. "cf,attr") if set --
    this script previously had no target list of its own and trained on whatever
    coverage_report.json marked ready, which is shared across every target ever
    prepped. That's what caused try05's job to silently also attempt cf/attr (added to
    TARGETS for the chat-reasoning pipeline) and time out. Without TARGETS_TO_RUN set,
    falls back to the old shared-report behavior."""
    report = json.load(open(os.path.join(DATA_DIR, "coverage_report.json")))
    _scope = os.environ.get("TARGETS_TO_RUN")
    if _scope:
        allowed = set(_scope.split(","))
        report = {name: r for name, r in report.items() if name in allowed}
    runnable = {name: r["status"] for name, r in report.items() if r["status"] in ("ready", "low_confidence")}
    skipped = {name: r["reason"] for name, r in report.items() if r["status"] == "skipped"}
    if skipped:
        print("Targets skipped at data-prep stage (not attempting to fine-tune these):")
        for name, reason in skipped.items():
            print(f"  {name}: {reason}")
    low_conf = [name for name, status in runnable.items() if status == "low_confidence"]
    if low_conf:
        print(f"Low-confidence targets (thin data, proceeding anyway -- their verdict below "
              f"will say so explicitly): {low_conf}")
    return runnable


def load_examples(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


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
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device
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
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        print(f"  epoch {epoch}: avg loss {total_loss / step:.4f}")
    model.eval()


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


def validate_target(model, tok, name, eval_set, confidence):
    ref = eval_set["reference"]
    ref_score = score(model, tok, ref["prompt"])
    print(f"[{name}] reference '{ref['name']}' predicted score: {ref_score!r} "
          f"(real label {ref['score']}, real potency={ref.get('potency_nM')} nM, held out of training -- "
          f"note this is the reference's absolute potency, not a claim it's a strong binder)")

    print(f"[{name}] easy negatives (unrelated drug classes -- weak test):")
    easy_scores = {}
    for c in eval_set["easy_negatives"]:
        s = score(model, tok, c["prompt"])
        easy_scores[c["name"]] = s
        print(f"    {c['name']}: {s!r}")

    print(f"[{name}] hard negatives (real near-miss structural analogs -- meaningful test):")
    hard_scores = {}
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

    if not beat_hard:
        verdict = "DO NOT TRUST for broader screening -- fails the hard-negative discrimination test"
    elif confidence == "low_confidence":
        verdict = ("WEAK PASS -- beat the negatives, but on very little training data. "
                   "This could be genuine signal or could be a small model memorizing a small "
                   "pool. Do not screen a full drug library on this alone; get more data first.")
    else:
        verdict = "PASS -- trust this target adapter for broader DrugBank-wide screening"
    print(f"[{name}] VALIDATION: beats easy negatives={beat_easy}, beats hard negatives={beat_hard}, "
          f"confidence={confidence}")
    print(f"[{name}] -> {verdict}\n")

    return {"reference_score": ref_score, "easy_scores": easy_scores, "hard_scores": hard_scores,
            "beats_easy": beat_easy, "beats_hard": beat_hard, "confidence": confidence}


if __name__ == "__main__":
    names = targets_to_run()
    if not names:
        raise SystemExit("No targets marked 'ready' in coverage_report.json -- run "
                          "prep_orphan_target_data.py first.")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # A fresh base-model load per target (rather than sharing one model across
    # multiple LoRA adapters via add_adapter/set_adapter) costs extra load time,
    # but a previous run using the shared-model approach produced one unexplained,
    # backwards result (a memorized reference drug scored worse than unrelated
    # negatives) that was never root-caused. Reloading per target is mechanically
    # simpler and removes any doubt about state leaking between adapters.
    summary = {}
    for name in names:
        result_dir = os.path.join(DATA_DIR, name)
        print(f"=== Loading base model for: {name} ===")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        model = get_peft_model(base_model, LORA_CONFIG, adapter_name=name)

        print(f"=== Training: {name} ===")
        examples = load_examples(os.path.join(result_dir, "train_blind.jsonl"))
        train_adapter(model, tok, examples)
        model.save_pretrained(os.path.join(result_dir, "adapter"), selected_adapters=[name])

        print(f"=== Validating: {name} ===")
        eval_set = json.load(open(os.path.join(result_dir, "eval_set.json")))
        summary[name] = validate_target(model, tok, name, eval_set, names[name])

        del model, base_model
        torch.cuda.empty_cache()

    with open(os.path.join(DATA_DIR, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== FINAL SUMMARY ===")
    for name, r in summary.items():
        print(f"{name}: beats_easy={r['beats_easy']}  beats_hard={r['beats_hard']}")
