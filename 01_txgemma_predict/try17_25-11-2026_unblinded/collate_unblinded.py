"""Collate every arm into one table: blinded model, unblinded model, blinded Tanimoto,
unblinded Tanimoto, random chance. One frozen pool, one rank convention, one file.

Tolerant of missing inputs -- runs under afterany, so a dead array task must produce a
row saying so rather than no report.

No beats_hard anywhere. See the README.
"""

import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_common as pc

BLIND_MODEL_DIR = os.path.join(pc.ORPHAN, "try16_equalized_pool")
UNBLIND_DIR = os.path.join(pc.ORPHAN, "try17_unblinded", "stepstop")
TAN_PATH = os.path.join(pc.ORPHAN, "try17_unblinded", "tanimoto_blind_vs_unblind.json")
OUT_ROOT = os.path.join(pc.ORPHAN, "try17_unblinded")

COLS = ["target", "ref_drug", "ref_real_label", "ref_potency_nM", "arm", "readout",
        "path_used", "seed", "rank_pool", "n_pool", "percentile_pool",
        "n_tied_with_ref", "predicted", "abs_label_error",
        "max_sim_to_actives", "trivial_flag", "status"]


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def main():
    tan_all = json.load(open(TAN_PATH)) if os.path.exists(TAN_PATH) else {}
    tan = tan_all.get("results", {})
    rand = tan_all.get("random_chance", {})

    rows = []
    for target in pc.ALL_TARGETS:
        ref = pc.reference_of(target)
        ev = json.load(open(os.path.join(pc.ORPHAN, target, "eval_set.json")))["reference"]
        base = {"target": target, "ref_drug": ref["name"],
                "ref_real_label": ref["real"], "ref_potency_nM": ev.get("potency_nM")}
        t = tan.get(target, {})

        # ---- Tanimoto arms
        for arm_name, key in (("tanimoto_blinded", "blinded"),
                              ("tanimoto_unblinded", "unblinded")):
            a = t.get(key, {})
            rows.append({**base, "arm": arm_name, "readout": "knn5",
                         "path_used": "", "seed": "",
                         "rank_pool": a.get("rank") or "",
                         "n_pool": a.get("n_actives") and tan_all.get("n_pool") or
                                   tan_all.get("n_pool", ""),
                         "percentile_pool": a.get("percentile") or "",
                         "n_tied_with_ref": a.get("n_tied_with_ref", ""),
                         "predicted": "", "abs_label_error": "",
                         "max_sim_to_actives": a.get("max_sim_to_actives", ""),
                         "trivial_flag": a.get("trivial_flag", ""),
                         "status": "ok" if a.get("rank") else
                                   "reference absent from pool_v1"})

        # ---- random chance
        rows.append({**base, "arm": "random_chance", "readout": "uniform",
                     "path_used": "", "seed": "",
                     "rank_pool": round(rand.get("expected_rank", 0)) or "",
                     "n_pool": tan_all.get("n_pool", ""),
                     "percentile_pool": rand.get("mean_percentile", ""),
                     "n_tied_with_ref": "", "predicted": "", "abs_label_error": "",
                     "max_sim_to_actives": "", "trivial_flag": "",
                     "status": f"empirical over {rand.get('trials', 0)} trials"})

        # ---- model arms
        for arm_name, root, fname in (
                ("model_blinded", BLIND_MODEL_DIR, f"{target}_pool_scores.json"),
                ("model_unblinded", UNBLIND_DIR, None)):
            if arm_name == "model_blinded":
                path = os.path.join(root, fname)
                if not os.path.exists(path):
                    rows.append({**base, "arm": arm_name, "readout": "", "path_used": "",
                                 "seed": "", "rank_pool": "", "n_pool": "",
                                 "percentile_pool": "", "n_tied_with_ref": "",
                                 "predicted": "", "abs_label_error": "",
                                 "max_sim_to_actives": "", "trivial_flag": "",
                                 "status": "MISSING: try16 task not finished"})
                    continue
                d = json.load(open(path))
                if not d.get("evaluable", False):
                    rows.append({**base, "arm": arm_name, "readout": "", "path_used": "",
                                 "seed": "", "rank_pool": "", "n_pool": "",
                                 "percentile_pool": "", "n_tied_with_ref": "",
                                 "predicted": "", "abs_label_error": "",
                                 "max_sim_to_actives": "", "trivial_flag": "",
                                 "status": f"NOT EVALUABLE: {d.get('reason', '')}"})
                    continue
                for tag, rec in sorted(d["results"].items()):
                    for readout in ("digit_exp", "greedy"):
                        r = rec[readout]
                        rows.append({**base, "arm": arm_name, "readout": readout,
                                     "path_used": rec["path_used"], "seed": rec["seed"],
                                     "rank_pool": r["rank"], "n_pool": rec["n_pool"],
                                     "percentile_pool": r["percentile"],
                                     "n_tied_with_ref": r["n_tied_with_reference"],
                                     "predicted": r["reference_score"],
                                     "abs_label_error": (
                                         abs(r["reference_score"] - ref["real"])
                                         if r["reference_score"] is not None else ""),
                                     "max_sim_to_actives": "", "trivial_flag": "",
                                     "status": "ok"})
            else:
                # try17 runs one array task per (target, seed), so each seed writes its
                # own shard. Merge them; fall back to the single-file name if a
                # 3-seeds-in-one-job run was used instead.
                import glob as _glob
                shards = sorted(_glob.glob(os.path.join(
                    root, target, "step_stopping_summary_seed*.json")))
                merged = {}
                for sh in shards:
                    merged.update(json.load(open(sh)).get("results", {}))
                sfile = os.path.join(root, target, "step_stopping_summary.json")
                marker = os.path.join(root, f"{target}_NOT_EVALUABLE.json")
                if os.path.exists(marker):
                    rows.append({**base, "arm": arm_name, "readout": "", "path_used": "",
                                 "seed": "", "rank_pool": "", "n_pool": "",
                                 "percentile_pool": "", "n_tied_with_ref": "",
                                 "predicted": "", "abs_label_error": "",
                                 "max_sim_to_actives": "", "trivial_flag": "",
                                 "status": "NOT EVALUABLE: reference absent from pool_v1"})
                    continue
                if not merged and not os.path.exists(sfile):
                    rows.append({**base, "arm": arm_name, "readout": "", "path_used": "",
                                 "seed": "", "rank_pool": "", "n_pool": "",
                                 "percentile_pool": "", "n_tied_with_ref": "",
                                 "predicted": "", "abs_label_error": "",
                                 "max_sim_to_actives": "", "trivial_flag": "",
                                 "status": "MISSING: unblinded task not finished"})
                    continue
                d = {"results": merged} if merged else json.load(open(sfile))
                for tag, rec in sorted(d["results"].items()):
                    for readout in ("digit_exp", "greedy"):
                        r = rec[readout]
                        rows.append({**base, "arm": arm_name, "readout": readout,
                                     "path_used": "stepstop_unblinded",
                                     "seed": rec["seed"],
                                     "rank_pool": r["rank"], "n_pool": r["n_pool"],
                                     "percentile_pool": r["percentile"],
                                     "n_tied_with_ref": r["n_tied_with_reference"],
                                     "predicted": r["reference"],
                                     "abs_label_error": r["abs_label_error"],
                                     "max_sim_to_actives": "", "trivial_flag": "",
                                     "status": "ok" if rec.get("reference_in_training")
                                               else "WARNING: reference not in training"})

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_csv = os.path.join(OUT_ROOT, "results_unblinded_pool_v1.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    # ----------------------------------------------------------------- SUMMARY.md
    L = ["# try17 — blinded vs unblinded, on the frozen equalized pool", "",
         f"Pool `pool_v1.json`, {tan_all.get('n_pool', '?')} approved drugs, sha256",
         "verified before every scoring run. Rank 1 = predicted most potent.",
         "**Lower percentile is better.** Random chance is 50%.", "",
         "`digit_exp` readout for the model arms; greedy is in the CSV but collapses.",
         "No hard-negative result is reported — see README.", "",
         "| target | drug | real | nM | model blind | model unblind | kNN blind | "
         "kNN unblind | random | leak |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for target in pc.ALL_TARGETS:
        rs = [r for r in rows if r["target"] == target]
        ref = pc.reference_of(target)
        ev = json.load(open(os.path.join(pc.ORPHAN, target,
                                         "eval_set.json")))["reference"]
        def pct(arm, readout):
            sel = [r["percentile_pool"] for r in rs
                   if r["arm"] == arm and r["readout"] == readout
                   and r["status"].startswith("ok") and r["percentile_pool"] != ""]
            m = mean_or_none([float(x) for x in sel])
            return f"{m:.2f}%" if m is not None else "—"
        t = tan.get(target, {})
        leak = "**TRIVIAL**" if t.get("blinded", {}).get("trivial_flag") else \
            str(t.get("blinded", {}).get("max_sim_to_actives", "—"))
        L.append(f"| {target} | {ref['name']} | {ref['real']} | {ev.get('potency_nM')} | "
                 f"{pct('model_blinded', 'digit_exp')} | "
                 f"{pct('model_unblinded', 'digit_exp')} | "
                 f"{pct('tanimoto_blinded', 'knn5')} | "
                 f"{pct('tanimoto_unblinded', 'knn5')} | "
                 f"{rand.get('mean_percentile', '—')}% | {leak} |")

    L += ["", "## How to read this", "",
          "**Rank is confounded with absolute potency.** A correct prediction for a weak",
          "binder ranks it low, and three references are weak: ambroxol 10,979 nM,",
          "lumacaftor 1,293 nM, tafamidis 839 nM. For those, a poor rank is the right",
          "answer and the number to read is `abs_label_error` in the CSV, not the rank.",
          "",
          "**The unblinded arms are ceilings, not results.** They are told the answer.",
          "A method that cannot rank the drug well even with the drug's own measurement",
          "in its training data has a representation problem, not a data problem.",
          "",
          "**`trivial_flag`** marks a reference with a Tanimoto >= 0.95 neighbour among",
          "the training actives, so even the blinded arm is partly reading its own",
          "answer back. Morgan-2 is stereo-blind, which is why fabry and nsiad flag at",
          "exactly 1.000: their diastereomers and enantiomer are indistinguishable from",
          "the drug in this representation."]

    out_md = os.path.join(OUT_ROOT, "SUMMARY.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
