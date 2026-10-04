"""
Stage-2 prototype: can the model apply an ALREADY-STATED necessary condition to rank a
realistic drug library, independent of whether it can derive that condition itself?

try02-try07 fused two questions into one number: (1) derive a necessary condition from a
mechanism description, and (2) apply it to rank candidates. A wrong final answer could
come from either, which made try04-try06 hard to diagnose. This isolates (2) by handing
the model the authored ground-truth condition and asking only for verdicts.

It also drops the 9-candidate multiple-choice format. That set was the true drug + 3
authored analogs + the 3 other folds' drugs + caffeine/aspirin, so "correct" meant a
1-of-9 pick against 2 obvious decoys -- far easier than the deployment task, and only
4 bits of signal across the whole LOTO. Here the pool is every approved drug in ChEMBL
(3417 unique) plus the 3 authored stereoisomer analogs spiked in as hard negatives, and
the metric is rank-of-true-drug, which is continuous and readable even at n=1 disease.

Scoring is a single forward pass per candidate -- logprob of "yes" vs "no" on a short
verdict prompt -- not a 2048-token generation. try07 took ~8 min per generated
completion; that approach cannot reach thousands of candidates, this can.

Defaults to the BASE model with no adapter: the honest baseline for "can bare TxGemma
apply a stated condition". Pass --adapter to compare a try06 LoRA.
"""
import argparse
import json
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "google/txgemma-27b-chat"
LIBRARY = "/work/hdd/bisa/sshah21/results/reasoning_screen/chembl_approved.json"
REASONING_DIR = "/work/hdd/bisa/sshah21/results/reasoning_chains"
OUT_DIR = "/work/hdd/bisa/sshah21/results/reasoning_screen"

PROMPT = """You are screening a candidate small molecule as a pharmacological chaperone, corrector, or stabilizer for a human protein.

A candidate must satisfy this necessary structural/physicochemical condition:
{condition}

Candidate compound (SMILES): {smiles}

Judge this candidate's real structure against the condition above. Does it satisfy the condition? Answer with one word, yes or no."""


def build_candidates(target):
    """Approved-drug library + the authored analogs for this target spiked in."""
    library = json.load(open(LIBRARY))
    chain = json.load(open(os.path.join(REASONING_DIR, f"{target}.json")))

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def canon(s):
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

    candidates, seen = [], set()
    for rec in library:
        c = rec.get("canonical_smiles") or canon(rec["smiles"])
        if not c or c in seen:
            continue
        seen.add(c)
        candidates.append({"name": rec["name"], "chembl_id": rec["chembl_id"],
                           "smiles": c, "source": "chembl_approved"})

    # Spike the authored hard negatives (stereoisomers / scaffold analogs of the true
    # drug) -- these are synthetic and absent from the approved library, but they are
    # the negatives try06 and try07 actually failed on, so the comparison needs them.
    prompt_smiles = dict(re.findall(r"(Compound \w+): (\S+)", chain["prompt"]))
    label_of = {v: k for k, v in chain["label_to_name"].items()}
    for name, label in label_of.items():
        if "analog" not in name:
            continue
        c = canon(prompt_smiles[label])
        if c in seen:
            continue
        seen.add(c)
        candidates.append({"name": name, "chembl_id": None, "smiles": c,
                           "source": "authored_analog"})

    return candidates, chain


def yes_no_token_ids(tok):
    """Token ids for yes/no in the surface forms the model may emit first."""
    def ids_for(words):
        out = set()
        for w in words:
            for variant in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.upper()):
                enc = tok.encode(variant, add_special_tokens=False)
                if enc:
                    out.add(enc[0])
        return sorted(out)
    return ids_for(["yes"]), ids_for(["no"])


@torch.no_grad()
def score_batch(model, tok, prompts, yes_ids, no_ids):
    """P(yes) vs P(no) at the first generated position, one forward pass per candidate."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     add_generation_prompt=True, tokenize=False)
             for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[:, -1, :].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    lp_yes = torch.logsumexp(logprobs[:, yes_ids], dim=-1)
    lp_no = torch.logsumexp(logprobs[:, no_ids], dim=-1)
    # Margin in log space: positive means the model leans yes. Ranking by this is
    # equivalent to ranking by P(yes | yes or no), but keeps resolution far out in
    # the tails where a normalized probability saturates at 0 or 1.
    return (lp_yes - lp_no).tolist(), lp_yes.tolist(), lp_no.tolist()


def report(scored, chain, target, batch_size, adapter):
    true_drug = chain["reference_drug"]
    ranked = sorted(scored, key=lambda r: -r["margin"])

    def rank_of(pred):
        """Rank by strictly-greater score, so ties do not inflate position."""
        for i, r in enumerate(ranked):
            if pred(r):
                better = sum(1 for o in ranked if o["margin"] > r["margin"])
                tied = sum(1 for o in ranked if o["margin"] == r["margin"])
                return i + 1, better + 1, tied, r
        return None, None, None, None

    n = len(ranked)
    print(f"\n{'=' * 70}\nSTAGE-2 SCREEN: {target} | library {n} candidates")
    print(f"model: {MODEL} | adapter: {adapter or 'NONE (base)'} | batch {batch_size}")
    print(f"condition handed to the model (authored ground truth):\n  {chain['necessary_condition'] if 'necessary_condition' in chain else chain['completion']['necessary_condition']}")

    _, tr, tied, rec = rank_of(lambda r: (r["name"] or "").upper() == true_drug.upper())
    if rec is None:
        print(f"\n!! true drug {true_drug} NOT FOUND in library -- cannot score")
    else:
        print(f"\nTRUE DRUG {true_drug}: rank {tr} / {n}  (top {100.0 * tr / n:.2f}%)"
              f" | margin {rec['margin']:+.3f} | {tied} candidates tied at this score")
        print(f"  recall@10  {'HIT' if tr <= 10 else 'miss'}"
              f" | recall@50  {'HIT' if tr <= 50 else 'miss'}"
              f" | recall@100 {'HIT' if tr <= 100 else 'miss'}")
        print(f"  enrichment over random: {n / (2.0 * tr):.1f}x")

    print("\nanchors (authored hard negatives and cross-target drugs):")
    anchors = [r["name"] for r in scored if r["source"] == "authored_analog"]
    anchors += ["LUMACAFTOR", "TAFAMIDIS", "MIGALASTAT", "CAFFEINE", "ASPIRIN"]
    for a in anchors:
        if a.upper() == true_drug.upper():
            continue
        _, ar, _, arec = rank_of(lambda r, a=a: (r["name"] or "").upper() == a.upper())
        if arec is not None:
            print(f"  {a:18s} rank {ar:5d} / {n}   margin {arec['margin']:+.3f}")

    # The calibration worry: on a curated 9-candidate set the model sees obvious decoys.
    # On a real library almost everything is a plausible organic molecule, so check
    # whether the verdicts still separate anything at all.
    margins = [r["margin"] for r in scored]
    n_yes = sum(1 for m in margins if m > 0)
    uniq = len(set(round(m, 6) for m in margins))
    print(f"\ncalibration: {n_yes}/{n} ({100.0 * n_yes / n:.1f}%) lean yes"
          f" | {uniq} distinct scores | margin range {min(margins):+.3f} .. {max(margins):+.3f}")

    print("\ntop 20 by margin:")
    for i, r in enumerate(ranked[:20], 1):
        tag = " <-- TRUE DRUG" if (r["name"] or "").upper() == true_drug.upper() else \
              " (authored analog)" if r["source"] == "authored_analog" else ""
        print(f"  {i:3d}. {r['margin']:+.3f}  {str(r['name'])[:44]:44s}{tag}")

    out = os.path.join(OUT_DIR, f"screen_{target}{'_adapter' if adapter else '_base'}.json")
    with open(out, "w") as f:
        json.dump({"target": target, "true_drug": true_drug, "n_candidates": n,
                   "adapter": adapter, "true_drug_rank": tr, "ranked": ranked}, f)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="fabry")
    ap.add_argument("--adapter", default=None, help="optional try06 LoRA dir")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on a subset")
    args = ap.parse_args()

    candidates, chain = build_candidates(args.target)
    if args.limit:
        # Keep the true drug and analogs in any smoke-test subset, or the metric is empty.
        keep = [c for c in candidates if c["source"] == "authored_analog"
                or (c["name"] or "").upper() == chain["reference_drug"].upper()]
        rest = [c for c in candidates if c not in keep]
        candidates = keep + rest[: max(0, args.limit - len(keep))]
    condition = chain["completion"]["necessary_condition"]
    print(f"scoring {len(candidates)} candidates for {args.target}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Decoder-only: pad left so the last position is the real end of every prompt.
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", dtype=torch.bfloat16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    yes_ids, no_ids = yes_no_token_ids(tok)
    print(f"yes token ids {yes_ids} | no token ids {no_ids}", flush=True)

    scored = []
    for i in range(0, len(candidates), args.batch_size):
        chunk = candidates[i:i + args.batch_size]
        prompts = [PROMPT.format(condition=condition, smiles=c["smiles"]) for c in chunk]
        margins, lp_yes, lp_no = score_batch(model, tok, prompts, yes_ids, no_ids)
        for c, m, ly, ln in zip(chunk, margins, lp_yes, lp_no):
            scored.append({**c, "margin": m, "lp_yes": ly, "lp_no": ln})
        if (i // args.batch_size) % 10 == 0:
            print(f"  scored {len(scored)}/{len(candidates)}", flush=True)

    report(scored, chain, args.target, args.batch_size, args.adapter)


if __name__ == "__main__":
    main()
