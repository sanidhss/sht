"""
Job C: collapse every try14 run into one results.csv plus a SUMMARY.md.

Reads whatever exists and reports what is missing rather than failing -- job C runs under
afterany, so it must produce a usable report even when some array tasks died or timed out.
That is deliberate: a chain left running unattended is most useful when the collator can
still tell you which targets finished and which did not.
"""
import csv
import json
import os

DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora_orphan"
TRY14_OUT = os.path.join(DATA_DIR, "try14_new_targets")
COLS = ["target", "path_used", "seed", "n_training", "n_val_distinct", "selected_step",
        "val_loss", "greedy_pct", "digit_exp_pct", "beats_hard",
        "selection_audit_agreed", "descriptor_knn_pct", "descriptor_ridge_pct",
        "descriptor_knn_beats_hard", "descriptor_ridge_beats_hard", "beat_baseline"]


def f1(x):
    return None if x is None else round(float(x), 1)


def f3(x):
    return None if x is None else round(float(x), 3)


def main():
    routing = {}
    rp = os.path.join(TRY14_OUT, "routing.json")
    if os.path.exists(rp):
        routing = json.load(open(rp))
    else:
        print(f"WARNING: no routing.json at {rp} -- job A did not finish. "
              f"Reporting whatever run output exists.")

    baseline = {}
    bp = os.path.join(TRY14_OUT, "descriptor_baseline.json")
    if os.path.exists(bp):
        baseline = json.load(open(bp))
    else:
        print("WARNING: no descriptor_baseline.json -- the baseline columns will be empty.")

    rows, missing, notes = [], [], []

    for target, r in sorted(routing.items()):
        path = r.get("path")
        if path == "none":
            notes.append(f"{target}: NOT RUN -- {r.get('reason', 'no reason recorded')}")
            rows.append({**{c: None for c in COLS}, "target": target,
                         "path_used": "none", "n_training": r.get("n_training", 0)})
            continue

        b = baseline.get(target, {})
        bk = (b.get("knn") or {}).get("enrichment") or {}
        br = (b.get("ridge") or {}).get("enrichment") or {}
        base_pcts = [p for p in (bk.get("percentile"), br.get("percentile")) if p is not None]
        best_base = min(base_pcts) if base_pcts else None

        if path == "stepstop":
            sp = os.path.join(TRY14_OUT, "stepstop", target, "step_stopping_summary.json")
            if not os.path.exists(sp):
                missing.append(f"{target} (stepstop): no summary at {sp}")
                continue
            d = json.load(open(sp))
            for tag, res in d["results"].items():
                ge = res["greedy"]["enrichment"] or {}
                de = res["digit_exp"]["enrichment"] or {}
                dpct = de.get("percentile")
                rows.append({
                    "target": target, "path_used": "stepstop", "seed": res.get("seed"),
                    "n_training": res.get("n_train_compounds"),
                    "n_val_distinct": res.get("n_distinct_val_molecules_in_curve"),
                    "selected_step": res.get("selected_step"),
                    "val_loss": f3(res.get("selected_val_loss")),
                    "greedy_pct": f1(ge.get("percentile")),
                    "digit_exp_pct": f1(dpct),
                    "beats_hard": res["digit_exp"]["beats_hard"],
                    "selection_audit_agreed": res.get("selection_check", {}).get("agrees"),
                    "descriptor_knn_pct": f1(bk.get("percentile")),
                    "descriptor_ridge_pct": f1(br.get("percentile")),
                    "descriptor_knn_beats_hard": (b.get("knn") or {}).get("beats_hard"),
                    "descriptor_ridge_beats_hard": (b.get("ridge") or {}).get("beats_hard"),
                    "beat_baseline": (None if (dpct is None or best_base is None)
                                      else dpct < best_base)})

        elif path == "crossval":
            sp = os.path.join(TRY14_OUT, "crossval", target,
                              f"{target}_crossval_summary.json")
            if not os.path.exists(sp):
                missing.append(f"{target} (crossval): no summary at {sp}")
                continue
            d = json.load(open(sp))
            audit = (d.get("selection_audit") or {}).get("agreed")
            for tag, res in d["results"].items():
                ge = res["greedy"]["enrichment"] or {}
                de = res["digit_exp"]["enrichment"] or {}
                dpct = de.get("percentile")
                rows.append({
                    "target": target, "path_used": "crossval", "seed": res.get("seed"),
                    "n_training": d.get("n_trainable") or r.get("n_training"),
                    # In the CV path every compound is a validation molecule exactly once,
                    # so the honest value for "distinct validation molecules" is the whole
                    # trainable set -- that is the entire reason this path exists.
                    "n_val_distinct": d.get("n_trainable") or r.get("n_training"),
                    "selected_step": d.get("stop_step_full"),
                    "val_loss": f3(d.get("chosen_mean_val")),
                    "greedy_pct": f1(ge.get("percentile")),
                    "digit_exp_pct": f1(dpct),
                    "beats_hard": res["digit_exp"]["beats_hard"],
                    "selection_audit_agreed": audit,
                    "descriptor_knn_pct": f1(bk.get("percentile")),
                    "descriptor_ridge_pct": f1(br.get("percentile")),
                    "descriptor_knn_beats_hard": (b.get("knn") or {}).get("beats_hard"),
                    "descriptor_ridge_beats_hard": (b.get("ridge") or {}).get("beats_hard"),
                    "beat_baseline": (None if (dpct is None or best_base is None)
                                      else dpct < best_base)})

    os.makedirs(TRY14_OUT, exist_ok=True)
    csv_path = os.path.join(TRY14_OUT, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c) for c in COLS})
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # ---- SUMMARY.md
    lines = ["# try14 results — new target set", ""]
    if missing:
        lines += ["## Runs that did not produce output", ""]
        lines += [f"- {m}" for m in missing] + [""]
    if notes:
        lines += ["## Targets not run", ""]
        lines += [f"- {n}" for n in notes] + [""]

    lines += ["## Per-seed results", "",
              "| target | path | seed | n_train | n_val_distinct | step | val_loss | "
              "greedy %ile | digit_exp %ile | beats_hard | audit | best baseline %ile | "
              "beat baseline? |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["path_used"] == "none":
            continue
        base_pcts = [p for p in (row["descriptor_knn_pct"], row["descriptor_ridge_pct"])
                     if p is not None]
        bb = min(base_pcts) if base_pcts else None
        lines.append(
            f"| {row['target']} | {row['path_used']} | {row['seed']} | {row['n_training']} | "
            f"{row['n_val_distinct']} | {row['selected_step']} | {row['val_loss']} | "
            f"{row['greedy_pct']} | {row['digit_exp_pct']} | {row['beats_hard']} | "
            f"{row['selection_audit_agreed']} | {bb} | {row['beat_baseline']} |")
    lines.append("")

    # ---- per-target verdicts
    lines += ["## Per-target verdict", ""]
    by_t = {}
    for row in rows:
        if row["path_used"] == "none":
            continue
        by_t.setdefault(row["target"], []).append(row)
    for t, rs in sorted(by_t.items()):
        ps = [r["digit_exp_pct"] for r in rs if r["digit_exp_pct"] is not None]
        nb = sum(1 for r in rs if r["beats_hard"])
        base_pcts = [p for p in (rs[0]["descriptor_knn_pct"], rs[0]["descriptor_ridge_pct"])
                     if p is not None]
        bb = min(base_pcts) if base_pcts else None
        spread = (max(ps) - min(ps)) if len(ps) > 1 else None
        beat = sum(1 for r in rs if r["beat_baseline"] is True)
        lines.append(f"**{t}** ({rs[0]['path_used']}, n_train={rs[0]['n_training']})")
        lines.append("")
        lines.append(f"- digit_exp enrichment across seeds: "
                     f"{[p for p in ps]}" + (f" (spread {spread:.1f})" if spread is not None else ""))
        lines.append(f"- leak-free hard-negative test: {nb}/{len(rs)} seeds passed")
        lines.append(f"- best non-LLM baseline: {bb} percentile; the fine-tune beat it on "
                     f"{beat}/{len(rs)} seeds")
        audits = [r["selection_audit_agreed"] for r in rs]
        if False in audits:
            lines.append(f"- **selection audit FAILED on "
                         f"{sum(1 for a in audits if a is False)}/{len(rs)} seeds** — the "
                         f"checkpoint choice is not resolved by the validation set, so "
                         f"treat the selected step as unreliable.")
        lines.append("")

    lines += ["## How to read this", "",
              "`digit_exp %ile` is the fraction of 500 MaxMin-diverse cross-target decoys "
              "predicted MORE potent than the reference drug, so **lower is better** and 50 "
              "is chance. `beats_hard` is the leak-free hard-negative test: does the "
              "reference outscore all three real structural analogs whose scaffolds were "
              "held out of training.", "",
              "The baseline columns are the number to clear. They come from "
              "`descriptor_baseline_predict.py`, fit on the same compounds under the same "
              "scaffold holdout and scored against the same decoys, so they are directly "
              "comparable. `beat_baseline` is true only when the fine-tune's digit_exp "
              "percentile is lower than the better of the two baselines. A fine-tune that "
              "cannot clear a Tanimoto k-NN lookup has not learned anything a lookup table "
              "does not already have.", "",
              "Decoy pools in try14 include the new targets' compounds, so these "
              "percentiles are comparable within try14 but not strictly against try11's.", ""]

    sp = os.path.join(TRY14_OUT, "SUMMARY.md")
    with open(sp, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {sp}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
