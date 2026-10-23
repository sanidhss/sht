"""
Distribution-based scoring -- is the output degeneracy a readout artifact?

NO TRAINING. Reuses the four already-saved adapters and try08's exact molecule set, and
changes only HOW a score is read out of the model.

THE HYPOTHESIS
--------------
Every score this project has produced came from:

    out = model.generate(**ids, max_new_tokens=8)

No sampling argument, so transformers defaults to do_sample=False -- plain greedy decoding,
argmax at each token. For a model that emits a number as text, greedy decoding returns the
MODAL output, not an estimate. That is exactly the shape of the degeneracy observed:

    Gaucher  791 on 102 of 200 novel compounds (51%)
    TTR      994 on ~40% of a 500-compound decoy set
    Fabry    630 on 49 of 200 (25%)
    CF       727 on 13 of 200 (6.5%)   <- the one that looks healthy

Those may not be predictions at all. They may be modes.

Supporting reason to think the capability is there and the readout is discarding it: try08
measured min_nonzero_gap = 1 for all four adapters, so the models demonstrably CAN produce
fine-grained values. If degeneracy were a hard capacity limit, that gap would be large.

So this run compares scoring methods on identical models and identical molecules. The only
variable is the readout.

    Method A  greedy                  -- the current method; must reproduce try08's numbers
    Method B  sampling mean (K, T)    -- sample K completions, parse, average
    Method C  digit-logit expectation -- deterministic smooth score from the token
                                         distribution, computed only if the tokenizer
                                         splits numbers per-digit (confirmed: it does,
                                         '791' -> ['7','9','1']). Digit WIDTH is taken from
                                         the greedy decode per prompt, because the scale
                                         runs to 1000 = four digits and a fixed width of 3
                                         would misread it by ~900.

Method A acting as a reproduction check matters: if greedy here does not match try08's
recorded values on the same molecules and adapters, something else is wrong and the
comparison is void. The script asserts nothing but prints the agreement rate.

WHY SAMPLING IS THE PRIMARY METHOD
----------------------------------
It makes no assumption about how the tokenizer splits numbers, so it cannot be silently
invalidated by a tokenization surprise. Method C is cheaper and deterministic but depends on
per-digit tokenization, so it is attempted defensively and skipped if the assumption fails.

Sampling also yields a free bonus: the standard deviation across samples is a per-molecule
UNCERTAINTY estimate. That is independently useful for ranking -- a candidate scored 400 +/- 15
is a different proposition from 400 +/- 200, and the current pipeline cannot tell them apart.

WHICH ADAPTERS, AND A LIMITATION
--------------------------------
The four original per-target adapters. These are the over-trained (3-epoch), leak-trained
models -- NOT try09's epoch-0 models, which were never saved (an oversight in try09: it
saved summaries but never called save_pretrained).

That turns out to be the right choice anyway: try08 measured the degeneracy on exactly these
adapters, so comparing greedy against distribution scoring on the same weights isolates the
readout change perfectly. It does mean the absolute numbers here inherit the over-training
problem, and the conclusion transfers to a properly-stopped model only by assumption.

WHAT WOULD COUNT AS SUCCESS
---------------------------
  Most-common-value frequency drops sharply (e.g. Gaucher's 51% -> single digits) and the
  reference's enrichment percentile holds or improves
      -> the degeneracy was a readout artifact. Big, cheap win: it also means every
         historical score in this project understated the model's resolution.

  Frequency drops but enrichment gets WORSE
      -> ties were being broken in the drug's favour by luck. The ranking signal is weaker
         than the greedy numbers implied. Useful, unwelcome.

  Frequency barely moves
      -> the degeneracy is real in the model, not the readout. Then the next lever is
         training-side (mid-epoch stopping, lower LR), not scoring.

Stated in advance so the result cannot be read selectively afterwards.
"""
import json
import os
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "google/txgemma-9b-predict"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
PROBE_DIR = os.path.join(DATA_DIR, "resolution_probe")
OUT_DIR = os.path.join(DATA_DIR, "distribution_scoring")

TARGETS = os.environ.get("TARGETS", "gaucher,attr,cf,fabry").split(",")
K_SAMPLES = int(os.environ.get("K_SAMPLES", "16"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))

# try08's greedy results, to verify Method A reproduces them.
TRY08_TOP = {"gaucher": (791, 102), "attr": (994, 57), "fabry": (630, 49), "cf": (727, 13)}
TRY08_DISTINCT_NOVEL = {"gaucher": 55, "attr": 94, "fabry": 46, "cf": 98}


def to_int(s):
    try:
        return int(str(s).strip())
    except (ValueError, AttributeError, TypeError):
        return None


def prompt_for(template, smiles):
    import re
    new, n = re.subn(r"(Drug SMILES: ).+", lambda m: m.group(1) + smiles, template, count=1)
    if n != 1:
        raise ValueError("SMILES substitution failed")
    return new


@torch.no_grad()
def score_greedy(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=8, do_sample=False)
    return to_int(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))


@torch.no_grad()
def score_sampled(model, tok, prompt, k, temperature):
    """K sampled completions -> mean / median / std. Tokenization-agnostic."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=8, do_sample=True, temperature=temperature,
                         top_p=0.95, num_return_sequences=k)
    plen = ids["input_ids"].shape[1]
    vals = [to_int(tok.decode(o[plen:], skip_special_tokens=True)) for o in out]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"mean": mean, "median": median, "std": var ** 0.5,
            "n_parsed": n, "n_unique": len(set(vals))}


def digit_token_map(tok):
    """{digit: token_id} if and only if each single digit is one token. Returns None
    otherwise, in which case Method C is skipped rather than silently wrong."""
    m = {}
    for d in range(10):
        ids = tok(str(d), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            return None
        m[d] = ids[0]
    return m


@torch.no_grad()
def score_digit_expectation(model, tok, prompt, dmap, n_digits=None, greedy_value=None):
    """Deterministic smooth score: walk the greedy prefix, and at each digit position take
    the probability-weighted expected digit over the renormalized digit distribution.

    This is an approximation -- E[d0]*100 + E[d1]*10 + E[d2] is not exactly E[number] --
    but it is smooth and monotone in the model's confidence, which is what breaking ties
    requires.

    n_digits MUST match the number the model is actually emitting. This tokenizer splits
    numbers per digit ('791' -> ['7','9','1']) but the scale runs to 1000, which is FOUR
    digits ('1','0','0','0'). Hardcoding 3 would read a true 1000 as ~100, a 900-point
    error, and try08 shows 1000 does occur in the novel stratum for TTR and Fabry. So the
    width is taken from the greedy decode for this same prompt when available, and only
    falls back to 3 otherwise.
    """
    if n_digits is None:
        n_digits = len(str(greedy_value)) if greedy_value is not None else 3
        n_digits = max(1, min(4, n_digits))
    ids = tok(prompt, return_tensors="pt").to(model.device)
    cur = ids["input_ids"]
    attn = ids["attention_mask"]
    total, place = 0.0, 10 ** (n_digits - 1)
    digit_ids = torch.tensor([dmap[d] for d in range(10)], device=cur.device)
    for _ in range(n_digits):
        out = model(input_ids=cur, attention_mask=attn)
        logits = out.logits[0, -1]
        dl = logits[digit_ids]
        probs = torch.softmax(dl.float(), dim=-1)
        exp_d = float(sum(p * d for d, p in enumerate(probs.tolist())))
        total += exp_d * place
        place //= 10
        nxt = digit_ids[int(torch.argmax(dl))].view(1, 1)
        cur = torch.cat([cur, nxt], dim=1)
        attn = torch.cat([attn, torch.ones_like(nxt)], dim=1)
    return total


def summarize(name, method, values):
    """values: list of numbers (floats ok)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"method": method, "n": 0}
    rounded = [round(v) for v in vals]
    counts = Counter(rounded)
    top_val, top_n = counts.most_common(1)[0]
    distinct = len(set(rounded))
    print(f"  [{name}/{method}] n={len(vals)} distinct={distinct} ({distinct/len(vals):.1%}) "
          f"most_common={top_val} x{top_n} ({top_n/len(vals):.1%})")
    return {"method": method, "n": len(vals), "n_distinct": distinct,
            "frac_distinct": distinct / len(vals), "top_value": top_val,
            "top_count": top_n, "top_frac": top_n / len(vals),
            "top5": counts.most_common(5)}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dmap = digit_token_map(tok)
    print(f"per-digit tokenization: {'yes -- Method C enabled' if dmap else 'NO -- Method C skipped'}")

    results = {}
    for target in TARGETS:
        adapter_dir = os.path.join(DATA_DIR, target, "adapter", target)
        raw_path = os.path.join(PROBE_DIR, f"{target}_raw_generations.json")
        if not (os.path.isdir(adapter_dir) and os.path.exists(raw_path)):
            print(f"[skip] {target}: missing adapter or try08 raw generations")
            continue

        # try08's EXACT molecules and its recorded greedy answers.
        raw = json.load(open(raw_path))
        novel = list(raw["diverse_novel"].keys())
        try08_greedy = {s: to_int(v) for s, v in raw["diverse_novel"].items()}

        eval_set = json.load(open(os.path.join(DATA_DIR, target, "eval_set.json")))
        template = eval_set["reference"]["prompt"]
        ref_smiles = eval_set["reference"]["smiles"]

        print(f"\n=== {target}: {len(novel)} novel molecules from try08, "
              f"K={K_SAMPLES} T={TEMPERATURE} ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="cuda:0", dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()

        greedy, sampled_mean, sampled_std, digit_exp = [], [], [], []
        agree = 0
        for i, s in enumerate(novel):
            p = prompt_for(template, s)
            g = score_greedy(model, tok, p)
            greedy.append(g)
            if g is not None and g == try08_greedy.get(s):
                agree += 1
            sd = score_sampled(model, tok, p, K_SAMPLES, TEMPERATURE)
            sampled_mean.append(sd["mean"] if sd else None)
            sampled_std.append(sd["std"] if sd else None)
            if dmap:
                digit_exp.append(score_digit_expectation(model, tok, p, dmap,
                                                         greedy_value=g))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(novel)} molecules", flush=True)

        n_cmp = sum(1 for s in novel if try08_greedy.get(s) is not None)
        print(f"  Method A reproduction vs try08: {agree}/{n_cmp} identical "
              f"({100*agree/max(n_cmp,1):.1f}%)")
        exp_top, exp_n = TRY08_TOP.get(target, (None, None))
        print(f"  try08 recorded: most_common={exp_top} x{exp_n}, "
              f"distinct={TRY08_DISTINCT_NOVEL.get(target)}")

        stats = {"greedy": summarize(target, "greedy", greedy),
                 "sampled_mean": summarize(target, f"sampled_mean(K={K_SAMPLES},T={TEMPERATURE})",
                                           sampled_mean)}
        if dmap:
            stats["digit_expectation"] = summarize(target, "digit_expectation", digit_exp)

        # Reference enrichment under each method -- same molecules, same model.
        ref_p = prompt_for(template, ref_smiles)
        ref_scores = {"greedy": score_greedy(model, tok, ref_p)}
        rs = score_sampled(model, tok, ref_p, K_SAMPLES, TEMPERATURE)
        ref_scores["sampled_mean"] = rs["mean"] if rs else None
        if dmap:
            ref_scores["digit_expectation"] = score_digit_expectation(
                model, tok, ref_p, dmap, greedy_value=ref_scores["greedy"])

        series = {"greedy": greedy, "sampled_mean": sampled_mean}
        if dmap:
            series["digit_expectation"] = digit_exp
        enrich = {}
        for m, vals in series.items():
            r = ref_scores.get(m)
            vv = [v for v in vals if v is not None]
            if r is None or not vv:
                enrich[m] = None
                continue
            better = sum(1 for v in vv if v < r)
            enrich[m] = {"rank": better, "n": len(vv), "percentile": 100.0 * better / len(vv),
                         "reference_score": r}
            print(f"  ENRICHMENT [{m}]: ref={r if isinstance(r,int) else round(r,1)} "
                  f"ranks {better}/{len(vv)} = {100*better/len(vv):.1f} percentile")

        mstd = [v for v in sampled_std if v is not None]
        if mstd:
            print(f"  per-molecule sampling std: mean {sum(mstd)/len(mstd):.1f}, "
                  f"min {min(mstd):.1f}, max {max(mstd):.1f}")

        results[target] = {
            "n_molecules": len(novel), "k_samples": K_SAMPLES, "temperature": TEMPERATURE,
            "method_A_reproduction_rate": agree / max(n_cmp, 1),
            "try08_recorded_top": TRY08_TOP.get(target),
            "stats": stats, "enrichment": enrich, "reference_scores": ref_scores,
            "mean_sampling_std": (sum(mstd) / len(mstd)) if mstd else None,
        }
        with open(os.path.join(OUT_DIR, f"{target}_scores.json"), "w") as f:
            json.dump({"novel": novel, "greedy": greedy, "sampled_mean": sampled_mean,
                       "sampled_std": sampled_std,
                       "digit_expectation": digit_exp if dmap else None}, f, indent=2)
        del model, base
        torch.cuda.empty_cache()
        with open(os.path.join(OUT_DIR, "distribution_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== FINAL SUMMARY: does distribution scoring break the degeneracy? ===")
    print(f"{'target':8s} {'method':34s} {'distinct%':>10s} {'top value freq':>15s} {'enrich %ile':>12s}")
    for t, r in results.items():
        for m, st in r["stats"].items():
            if not st.get("n"):
                continue
            e = r["enrichment"].get(m)
            ep = f"{e['percentile']:.1f}" if e else "n/a"
            print(f"{t:8s} {st['method'][:34]:34s} {st['frac_distinct']:9.1%} "
                  f"{st['top_value']:>7} x{st['top_frac']:<6.1%} {ep:>12s}")
    print("\nPre-registered reading (see this try's README):")
    print("  top-value freq drops sharply AND enrichment holds/improves -> readout artifact; cheap win")
    print("  top-value freq drops BUT enrichment worsens -> ties were breaking in the drug's favour by luck")
    print("  top-value freq barely moves -> degeneracy is real in the model; next lever is training-side")
