# Try 06 — Killed by a directory reorg that happened mid-queue

**Date:** 22-09-2026
**Script:** cleaned-up version of `txgnn_fabry.py` (helper-function refactor, both
diseases parameterized via a `DISEASE` env var) — this exact version superseded by later
edits, not separately preserved.
**Jobs:** 22282315 (FAILED, Fabry), 22282316 (FAILED, Gaucher) — both failed in **14
seconds**, i.e. before any real work happened.

## What happened / why it failed
Not a science bug at all — pure infrastructure. Both jobs sat in the SLURM queue for
~24 hours due to cluster congestion. While queued, the `scripts/` directory got
reorganized (`.py` files moved into `py_scripts/`, `.sb` files into `sbatch/`), but the
already-queued `.sb` scripts still referenced the old flat path. When they finally got a
GPU slot, both failed instantly: `can't open file '.../scripts/txgnn_fabry.py': No such
file or directory`.

Caught and fixed by updating the `.sb` path — and, since the same reorg had happened to
every other `.sb` file in the project, all of them were checked and fixed at the same
time (not just these two).
