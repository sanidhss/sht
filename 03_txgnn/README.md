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
8. **try09** — root-caused the attention bug precisely (a DGL minibatch/block sampling
   quirk for same-type relations) and fixed it by skipping `pretrain()` when
   `attention=True`, verified safe via `finetune()`'s own source. Training itself
   worked under this fix (both diseases trained to completion) but both jobs crashed
   at the final **test-evaluation** step: `KeyError: ('drug', 'contraindication',
   'disease')` in `evaluate_fb()`.
9. **try10 (current)** — verified (against PrimeKG directly, not assumed) that the
   `evaluate_fb` crash was a library gap, not a rare-disease data gap, and patched it.
   Resubmission trained and evaluated fully for the first time (22454275/22454276),
   but hit a second bug in this project's own `rank_of()`: it assumed
   `out['prediction']` is keyed by TxGNN's internal disease index, but it's actually
   keyed by PrimeKG's raw disease id (a different value/format per disease, one of
   them a merged multi-id string). Also caught and fixed a path-drift bug where
   `txgnn_fabry.sb` was still silently pointing at try09's script. Computed the real
   result directly from the already-saved `eval.pkl` files rather than waiting on a
   resubmission, then resubmitted for an official clean log (22544783/22544784).

## Result (computed 29-09-2026, from a real completed training+eval run)
**Fabry: migalastat ranks a mean of 17.7 / 7,957 drugs** (12th/22nd/19th across 3
seeds) — strong, consistent, top-0.2%. A large improvement over try03's early
`attention=False` result (2,863rd).

**Gaucher: weak and mixed per-seed**, but the picture changes with ensembling. Best
per-seed drug was miglustat (mean rank 1251, wildly seed-variable: 7th to 3724th).
The 3 enzyme-replacement biologics (alglucerase, imiglucerase, taliglucerase alfa) all
score exactly 0.0 identically across all 3 seeds — a likely missing-feature artifact,
not yet root-caused.

11. **try11 (current)** — root-caused the Gaucher seed variance: checkpoint selection
    (`finetune()`) picks the epoch with the best *macro* AUROC across all 3
    relation types in validation (contraindication/indication/off-label), not the
    indication-specific AUROC that's actually tested — a checkpoint can win on the
    other two relations while being mediocre at indication, and which one wins is
    seed-dependent noise. Confirmed for free by ensembling try10's 3 already-trained
    seeds (averaging raw scores, no retraining): **miglustat's ensembled rank is
    17/7,957** — recovering a Fabry-caliber result that was just being obscured by
    per-seed checkpoint noise. The 3 biologics stayed flat at ~6070 even after
    ensembling, confirming that artifact is structurally different (unaffected by
    noise-cancellation). Patched checkpoint selection to use indication-specific
    AUROC instead of the 3-relation macro average, and submitted a full retrain
    (22544937 fabry, 22544938 gaucher) under the new criterion.
