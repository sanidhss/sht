# Project: TxGNN (PrimeKG knowledge-graph link prediction)

**Question:** Can a graph-based model (disease/gene/drug relationships in PrimeKG, no
chemistry at all) correctly identify the right drug for Fabry/Gaucher while being
provably blind to the true answer during training?

## Arc (9 tries, the longest debugging arc of the three projects)
1. **try01-02** — old environment (`envs/txgnn`, dgl 0.9.1): two different crashes, one
   from a wrong disease-index bug (see try03), one from a DGL API version mismatch after
   trying to fix the first by upgrading dgl in-place. Motivated building a fresh,
   modern environment (`envs/txgnn2`) instead of continuing to patch the old one.
2. **try03** — new environment works mechanically; **Fabry trained fully and gave a
   real, disappointing result** (migalastat ranked 2,863rd of 7,957 drugs) — not a bug,
   a genuine finding. Root-caused the crash affecting Gaucher: `disease_idxs=31384` was
   PrimeKG's raw global node-table row number, not TxGNN's internal per-type index
   (correct value: 543).
3. **try04** — 3 jobs submitted and cancelled mid-fix, no wasted compute.
4. **try05** — correct index in place; new bug found (Gaucher genuinely has zero
   `contraindication`/`off-label` edges in PrimeKG, and the library crashes rather than
   handling that gracefully). Patched.
5. **try06** — pure infrastructure failure: a directory reorg happened while jobs sat in
   a 24h queue, breaking hardcoded paths. Fixed across every `.sb` file in the project.
6. **try07** — added `attention=True` (to fix a real dilution problem: migalastat's one
   meaningful edge drowned among 770 unrelated drug-drug edges) + multi-seed. Found that
   `complex_disease`'s blindness for a given disease is seed-dependent luck (seed=1
   leaks Fabry, seed=42 doesn't) — switched to the structurally-guaranteed
   `disease_eval` split. Also found a brand-new TxGNN library bug in the attention layer.
7. **try08** — confirmed the attention bug affects both diseases identically once both
   use `disease_eval`. Patched the library's error handling to reveal the real
   exception (previously swallowed into a bare `ValueError`).
8. **try09 (current)** — root-caused the attention bug precisely (a DGL minibatch/block
   sampling quirk for same-type relations) and fixed it by skipping `pretrain()` when
   `attention=True`, verified safe via `finetune()`'s own source. Not yet run under this
   fix.

## Current state
Zero results yet under `attention=True` — every attempt so far has been infrastructure
or library bugs, not a completed science run. try09's fix is the first one with a real
chance of producing an actual ranking for both diseases; resubmission is next.
