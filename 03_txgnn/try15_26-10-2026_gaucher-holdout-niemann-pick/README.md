# try15 — Gaucher with Niemann-Pick also held out

## Question
try14 found miglustat → *Niemann-Pick disease* (idx 1490) in Gaucher's training graph,
2 hops from Gaucher (4 shared disease_disease neighbours, incl. *sphingolipidosis*).
It's legitimate knowledge (miglustat is approved for NPC), not a leaked label, but
it's the likeliest driver of miglustat's ensembled rank of 17/7,957. Does that rank
survive without it?

## Design
Identical to try11/try13 (same model config, 3 seeds, seed-ensembled rank) except the
holdout is `[16430 (Gaucher), 1490 (Niemann-Pick)]`. The only drug edge on any
Niemann-Pick node in PrimeKG is miglustat → Niemann-Pick indication (*Niemann-Pick
disease type C* has none), so exactly **one training edge** is removed: a clean
single-variable ablation. Results go in `.../txgnn_v2_indication_selection/gaucher_no_np/`.
That's a new dir on purpose: the script skips training when a checkpoint exists, and
try11's Gaucher checkpoints must not be reused.

Files: `txgnn_gaucher_no_np.py` (try13 script, config block changed only; logic
`diff`-identical), `txgnn_gaucher_no_np.sb` (invokes this folder's script; checked),
`dryrun.py` / `dryrun.out`.

## Status
- Dry run (26-10-2026): split built, `check_blind` OK. Re-running try14's
  neighbour-ring audit on this split finds **no** remaining graph-close indication
  paths for miglustat, eliglustat, velaglucerase alfa or voglibose.
- **Submitted 26-10-2026 as job 23210548** (12h). Results pending.

## How to read the result
- Miglustat stays near 17 → the Gaucher result doesn't depend on the sibling indication.
- Miglustat falls to hundreds or thousands → Gaucher's "success" was mostly the treated
  sibling, and the thesis should say so.
