"""STEP 3 -- collate the adapter arm and the Tanimoto arm into results_pool_v1.csv.

Tolerant of missing inputs by design: this runs under afterany, so it must produce a
usable report even when an array task died. A target with no scores file is written out
with its reason rather than dropped silently.

beats_hard is deliberately ABSENT from this file. See the README -- try14 established
that hard-negative SELECTION is broken on 3 of 4 new targets, so the flag does not
measure what its name says and would be actively misleading next to an honest pool rank.
"""

import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_common as pc

COLS = ["target", "ref_drug", "path_used", "seed", "readout",
        "rank_pool", "n_pool", "percentile_pool",
        "n_tied_with_ref", "n_distinct_scores",
        "tanimoto_rank_pool", "tanimoto_percentile_pool",
        "max_sim_to_actives", "trivial_flag", "status"]


def main():
    tan_path = os.path.join(pc.OUT_ROOT, "tanimoto_pool_v1.json")
    tan = json.load(open(tan_path))["results"] if os.path.exists(tan_path) else {}

    rows, missing = [], []
    for target in pc.ALL_TARGETS:
        t = tan.get(target, {})
        t_rank = t.get("rank")
        t_pct = t.get("percentile")
        max_sim = t.get("max_sim_to_actives")
        trivial = t.get("trivial_flag")
        ref_name = t.get("reference", "")

        path = os.path.join(pc.OUT_ROOT, f"{target}_pool_scores.json")
        if not os.path.exists(path):
            missing.append((target, "no scores file -- array task did not finish"))
            rows.append({"target": target, "ref_drug": ref_name, "path_used": "",
                         "seed": "", "readout": "", "rank_pool": "", "n_pool": "",
                         "percentile_pool": "", "n_tied_with_ref": "",
                         "n_distinct_scores": "",
                         "tanimoto_rank_pool": t_rank or "",
                         "tanimoto_percentile_pool": t_pct or "",
                         "max_sim_to_actives": max_sim, "trivial_flag": trivial,
                         "status": "MISSING: array task did not finish"})
            continue

        d = json.load(open(path))
        if not d.get("evaluable", False):
            reason = d.get("reason", "not evaluable")
            missing.append((target, reason))
            rows.append({"target": target, "ref_drug": d.get("reference", ref_name),
                         "path_used": "", "seed": "", "readout": "", "rank_pool": "",
                         "n_pool": "", "percentile_pool": "", "n_tied_with_ref": "",
                         "n_distinct_scores": "",
                         "tanimoto_rank_pool": t_rank or "",
                         "tanimoto_percentile_pool": t_pct or "",
                         "max_sim_to_actives": max_sim, "trivial_flag": trivial,
                         "status": f"NOT EVALUABLE: {reason}"})
            continue

        for tag, rec in sorted(d["results"].items()):
            for readout in ("digit_exp", "greedy"):
                r = rec[readout]
                rows.append({
                    "target": target, "ref_drug": d["reference"],
                    "path_used": rec["path_used"], "seed": rec["seed"],
                    "readout": readout,
                    "rank_pool": r["rank"], "n_pool": rec["n_pool"],
                    "percentile_pool": r["percentile"],
                    "n_tied_with_ref": r["n_tied_with_reference"],
                    "n_distinct_scores": r["n_distinct_scores"],
                    "tanimoto_rank_pool": t_rank, "tanimoto_percentile_pool": t_pct,
                    "max_sim_to_actives": max_sim, "trivial_flag": trivial,
                    "status": "ok",
                })

    os.makedirs(pc.OUT_ROOT, exist_ok=True)
    out_csv = os.path.join(pc.OUT_ROOT, "results_pool_v1.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    # ------------------------------------------------------------------ SUMMARY.md
    lines = ["# try16 — adapters vs Tanimoto 5-NN on the frozen equalized pool", "",
             "Pool `pool_v1.json`, 1,933 approved drugs, sha256 verified before every",
             "scoring run. Rank 1 = predicted most potent. **Lower is better**; 50% is",
             "chance. The reference drug is one of the 1,933, not spiked in.", "",
             "`beats_hard` is deliberately absent — see README.", ""]

    if missing:
        lines += ["## Targets not evaluated", ""]
        for t, why in missing:
            lines.append(f"- **{t}**: {why}")
        lines.append("")

    lines += ["## Per-target", ""]
    lines += ["| target | drug | readout | path | pool rank (mean) | pool %ile | "
              "Tanimoto rank | Tanimoto %ile | model beats Tanimoto | leak |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for target in pc.ALL_TARGETS:
        rs = [r for r in rows if r["target"] == target and r["status"] == "ok"]
        if not rs:
            continue
        t = tan.get(target, {})
        for readout in ("digit_exp", "greedy"):
            for path_used in sorted({r["path_used"] for r in rs}):
                sel = [r for r in rs if r["readout"] == readout
                       and r["path_used"] == path_used]
                if not sel:
                    continue
                ranks = [r["rank_pool"] for r in sel if r["rank_pool"]]
                pcts = [r["percentile_pool"] for r in sel if r["percentile_pool"]]
                if not ranks:
                    continue
                mr = statistics.mean(ranks)
                mp = statistics.mean(pcts)
                tp = t.get("percentile")
                beat = "—" if tp is None else ("**yes**" if mp < tp else "no")
                leak = "**TRIVIAL**" if t.get("trivial_flag") else \
                    f"{t.get('max_sim_to_actives')}"
                lines.append(
                    f"| {target} | {sel[0]['ref_drug']} | {readout} | {path_used} | "
                    f"{mr:.0f} | {mp:.2f}% | {t.get('rank')} | "
                    f"{tp:.2f}% | {beat} | {leak} |" if tp is not None else
                    f"| {target} | {sel[0]['ref_drug']} | {readout} | {path_used} | "
                    f"{mr:.0f} | {mp:.2f}% | — | — | — | {leak} |")

    lines += ["", "## How to read this", "",
              "`percentile_pool` is the reference drug's rank among 1,933 real approved",
              "drugs, so it answers the deployment question directly: how deep a",
              "shortlist must a screen carry before the right drug is in it.",
              "",
              "`n_tied_with_ref` matters for the greedy readout. try14 measured 54",
              "distinct greedy values across 500 decoys, so greedy ties in bulk and its",
              "rank is reported BEST-CASE — every tie broken in the reference's favour.",
              "A greedy rank with hundreds of ties is not a real rank. digit_exp",
              "effectively never ties.",
              "",
              "`trivial_flag` is set when the reference has a Tanimoto >= 0.95 neighbour",
              "among the training actives. The reference itself is held out, but a",
              "near-identical analogue may not be, and a 5-NN lookup then reads its own",
              "answer back. Any row flagged TRIVIAL must not be reported as a result."]

    out_md = os.path.join(pc.OUT_ROOT, "SUMMARY.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
