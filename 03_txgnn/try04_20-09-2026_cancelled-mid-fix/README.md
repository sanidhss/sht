# Try 04 — Submitted and cancelled while iterating on the index fix

**Date:** 20/21-09-2026
**Script:** not preserved — intermediate states between try03's fix and try05's actual
run.
**Jobs:** 22248544 (CANCELLED), 22281535 (CANCELLED), 22281720 (CANCELLED) — **none of
these ever started running**, so there is no log/outcome to document beyond the
cancellation itself.

## Why these were cancelled rather than left to run
Each was cancelled *before* it started (still `PD` in the queue) upon discovering a
further problem that made the already-queued version not worth running:
- 22248544: cancelled after finding the missing-edge-type bug that would affect it (see
  try05) — patched the library first rather than waste a queue slot on a run known to
  fail the same way.
- 22281535 / 22281720: cancelled after deciding to add `attention=True` + multi-seed,
  which needed a bigger time budget and a rewritten script — no point letting an
  outdated version start.

No wasted GPU time — all three were caught and cancelled while still pending.
