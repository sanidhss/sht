"""STEP 1 -- score the frozen equalized pool with the SAVED adapters. Inference only.

No training, no optimizer, no gradients, no checkpoint selection, and nothing is
refitted. Each adapter is loaded from disk exactly as step-level stopping (try11/try14)
or cross-validation (try12) saved it. If an adapter is absent the target is reported as
missing and skipped -- never refitted, because a refit would not be the same model the
earlier tries reported.

Readout is try11's `score_both`, copied verbatim: greedy AND digit-logit expectation
from ONE forward pass, so the two readouts cannot drift apart and the numbers stay
comparable with try11/try14/try15. Digit width is taken from the greedy decode rather
than fixed at 3 -- the try10 bug, which misread 1000 by ~900.

One base model load per array task, with adapters swapped in and out around it.
"""

import json
import os
import re
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_common as pc

MODEL = "google/txgemma-9b-predict"
SEEDS = [0, 1, 2]
ORPHAN = pc.ORPHAN

# Where each path's adapters live. `tag` is the adapter_name used at save time, which is
# also the subdirectory PEFT wrote inside <...>_adapter/.
STEPSTOP_DIRS = {
    "gaucher": f"{ORPHAN}/step_stopping/gaucher",
    "cf": f"{ORPHAN}/step_stopping/cf",
    "fabry": f"{ORPHAN}/step_stopping/fabry",
    "attr": f"{ORPHAN}/step_stopping/attr",
    "cel": f"{ORPHAN}/try14_new_targets/stepstop/cel",
    "mastocytosis": f"{ORPHAN}/try14_new_targets/stepstop/mastocytosis",
    "nsiad": f"{ORPHAN}/try14_new_targets/stepstop/nsiad",
    "tyrosinemia": f"{ORPHAN}/try14_new_targets/stepstop/tyrosinemia",
}
CROSSVAL_DIRS = {"fabry": f"{ORPHAN}/fabry_crossval"}


def adapters_for(target):
    """[(path_used, seed, adapter_dir)] for every saved adapter this target has."""
    found = []
    d = STEPSTOP_DIRS.get(target)
    if d:
        for s in SEEDS:
            tag = f"{target}_seed{s}"
            p = os.path.join(d, f"{tag}_adapter", tag)
            if os.path.isdir(p):
                found.append(("stepstop", s, p))
    d = CROSSVAL_DIRS.get(target)
    if d:
        for s in SEEDS:
            tag = f"{target}_cv_seed{s}"
            p = os.path.join(d, f"{tag}_adapter", tag)
            if os.path.isdir(p):
                found.append(("crossval", s, p))
    return found


# ------------------------------------------------- readout, verbatim from try11/try15
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
    """Greedy AND digit-logit expectation from ONE generate call."""
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


def main():
    target = os.environ.get("TARGET", "").strip()
    if not target:
        print("FATAL: TARGET is not set. Refusing to guess -- the try05 scope-coupling "
              "bug came from an implicit default.", flush=True)
        sys.exit(2)
    if target not in pc.ALL_TARGETS:
        print(f"FATAL: unknown target {target!r}; valid: {pc.ALL_TARGETS}", flush=True)
        sys.exit(2)

    pool = pc.verify_pool()
    structures = pc.pool_structures(pool)
    ref = pc.reference_of(target)

    pool_smis = [c for c, _ in structures]
    pool_names = [r.get("name", "") for _, r in structures]
    if ref["smiles"] not in set(pool_smis):
        print(f"{target}: reference {ref['name']!r} is NOT in pool_v1 -- cannot be "
              f"ranked on this pool. Reported as not evaluable; no substitute spiked "
              f"in, because every arm must score the identical frozen pool.", flush=True)
        os.makedirs(pc.OUT_ROOT, exist_ok=True)
        with open(os.path.join(pc.OUT_ROOT, f"{target}_pool_scores.json"), "w") as fh:
            json.dump({"target": target, "reference": ref["name"],
                       "evaluable": False,
                       "reason": "reference absent from pool_v1",
                       "results": {}}, fh, indent=2)
        return

    adapters = adapters_for(target)
    if not adapters:
        print(f"{target}: NO SAVED ADAPTER FOUND -- skipping, not refitting.", flush=True)
        sys.exit(3)
    print(f"{target}: {len(adapters)} adapters -> "
          f"{[(p, s) for p, s, _ in adapters]}", flush=True)

    ref_idx = pool_smis.index(ref["smiles"])
    tok = AutoTokenizer.from_pretrained(MODEL)
    dmap = digit_token_map(tok)
    if dmap is None:
        print("FATAL: digits are not single tokens; digit_exp readout impossible.")
        sys.exit(4)

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="cuda:0", dtype=torch.bfloat16)
    base.eval()
    print(f"base model loaded in {time.time() - t0:.0f}s", flush=True)

    out_all = {}
    for path_used, seed, adir in adapters:
        tag = f"{path_used}_seed{seed}"
        t1 = time.time()
        model = PeftModel.from_pretrained(base, adir)
        model.eval()
        # Assert the adapter actually attached. A silently-unapplied adapter would make
        # this run a second copy of try15's zero-shot and nobody would notice.
        n_lora = sum(1 for n, _ in model.named_modules() if "lora_A" in n)
        if n_lora == 0:
            print(f"FATAL: {adir} produced no LoRA layers", flush=True)
            sys.exit(5)

        greedy, digit = [], []
        for i, smi in enumerate(pool_smis):
            g, d = score_both(model, tok, prompt_for(ref["prompt"], smi), dmap)
            greedy.append(g)
            digit.append(d)
            if (i + 1) % 400 == 0:
                print(f"    {tag}: {i + 1}/{len(pool_smis)} "
                      f"({time.time() - t1:.0f}s)", flush=True)

        rec = {"path_used": path_used, "seed": seed, "adapter": adir,
               "n_pool": len(pool_smis), "n_lora_layers": n_lora}
        for name, scores in (("greedy", greedy), ("digit_exp", digit)):
            rank, tied = pc.rank_in_pool(scores[ref_idx], scores)
            n_distinct = len({v for v in scores if v is not None})
            rec[name] = {
                "reference_score": scores[ref_idx],
                "rank": rank,
                "percentile": round(100.0 * rank / len(pool_smis), 3) if rank else None,
                "n_tied_with_reference": tied,
                "n_distinct_scores": n_distinct,
                "n_unscored": sum(1 for v in scores if v is None),
            }
            print(f"  {tag} {name:9} rank {rank}/{len(pool_smis)} "
                  f"({100.0 * rank / len(pool_smis):.2f}%)  tied {tied}  "
                  f"distinct {n_distinct}", flush=True)
        rec["scores"] = {"greedy": greedy, "digit_exp": digit}
        out_all[tag] = rec

        # Detach the adapter before the next one so weights cannot accumulate.
        base = model.unload()
        del model
        torch.cuda.empty_cache()
        print(f"  {tag} done in {time.time() - t1:.0f}s", flush=True)

    os.makedirs(pc.OUT_ROOT, exist_ok=True)
    payload = {"target": target, "reference": ref["name"],
               "reference_real": ref["real"], "evaluable": True,
               "pool_sha256_verified": True, "n_pool": len(pool_smis),
               "reference_pool_index": ref_idx,
               "pool_names": pool_names, "pool_smiles": pool_smis,
               "results": out_all}
    out = os.path.join(pc.OUT_ROOT, f"{target}_pool_scores.json")
    with open(out, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {out}  (total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
