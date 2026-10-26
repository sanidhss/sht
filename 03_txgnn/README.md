# Project: TxGNN (PrimeKG knowledge-graph link prediction)

**Question:** Can a graph-based model (disease/gene/drug relationships in PrimeKG, no
chemistry at all) correctly identify the right drug for Fabry/Gaucher while being
provably blind to the true answer during training?

## Arc (22 tries, the longest debugging arc of the three projects)
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

## Adopted mitigation for seed variance
**Ensembling (averaging raw scores across the 3 seeds)**, not a training/selection
fix — try11 showed the variance is inherent embedding-space stochasticity, not a bug.
This is the validated path forward for reporting results on any target going
forward: train 3 seeds, average scores, report the ensembled rank, not any single
seed's.

## Result (computed 29-09-2026, from a real completed training+eval run)
**Fabry: migalastat ranks a mean of 17.7 / 7,957 drugs** (12th/22nd/19th across 3
seeds) — strong, consistent, top-0.2%. A large improvement over try03's early
`attention=False` result (2,863rd).

**Gaucher: weak and mixed per-seed**, but the picture changes with ensembling. Best
per-seed drug was miglustat (mean rank 1251, wildly seed-variable: 7th to 3724th).
The 3 enzyme-replacement biologics (alglucerase, imiglucerase, taliglucerase alfa) all
score exactly 0.0 identically across all 3 seeds — root-caused in try12: their only
PrimeKG edge is the held-out Gaucher indication, so they are isolated nodes in the
training graph and get a zero embedding (no self-loop). Not evaluable; excluded from
the Gaucher reference set (now 4 drugs).

11. **try11** — hypothesized the Gaucher seed variance was a checkpoint-selection
    artifact (macro AUROC across 3 relation types vs. the indication-specific AUROC
    actually tested) and patched it. **Result: hypothesis rejected.** The retrain
    under the corrected criterion gave essentially identical numbers to try10
    (migalastat 17.3 vs 17.7 mean rank; miglustat 1233.7 vs 1251.3, same 7/25-ish vs.
    3669-3724 per-seed spread). Checked seed 1 vs. seed 2's actual validation AUROC
    trajectories: seed 1 (0.8079) is comparable to, even slightly better than, seed 2
    (0.7960) — if checkpoint selection were the cause, the "bad" seed should show
    worse validation metrics, and it doesn't. Real conclusion: this is inherent
    stochasticity in the learned embedding space itself (aggregate indication AUROC,
    averaged across hundreds of diseases, doesn't predict where one specific
    (drug, disease) pair lands), not a fixable training bug.
12. **try12** — root-caused the exact-0.0 scores (CPU-only diagnostic, no training).
    Every exact-0.0 drug, in all 6 disease×seed runs, has exactly one edge in all of
    PrimeKG (a single indication) and zero edges in the training split; the
    zero-score set and zero-train-degree set are identical. The RGCN layers have no
    self-loop, so an isolated node gets an all-zero representation and DistMult
    scores it exactly 0. The "biologics lack chemistry features" guess was wrong —
    velaglucerase alfa (same drug class, one extra drug_protein edge) scores normally.
    The 3 Gaucher ERTs are structurally unscoreable under a blind split; reported as
    not evaluable, not as misses. No library change.
13. **try13** — extends to CF and TTR. Precheck (try12's degree check) flags dornase
    alfa (CF) as isolated/unscoreable in advance. TTR needed care: PrimeKG's
    hereditary/wild-type ATTR nodes have no indications; tafamidis is labelled only
    on ATTRV122I plus 3 generic/sibling amyloidosis nodes, so all are held out
    together to prevent sibling-node leakage. Jobs 22996974 (CF) / 22996975 (TTR)
    completed 20-10-2026. **Result: both weak.**
    CF: the 4 CFTR modulators cluster at ensemble ranks 1,208–1,652/7,957 (top 15–21%,
    above chance every seed, ~70–100x worse than Fabry). TTR: tafamidis 974 at the one
    labelled node (ATTRV122I); hereditary ATTR node is near-empty (4 edges) and gives
    no signal. Shared drug-protein-disease target exists in all cases (incl. Fabry and
    the failures), so it doesn't explain the gap; neighbour dilution is an untested
    hypothesis. Library AUROC is restricted to indication-bearing drugs and is not
    comparable to the full-library rank.
14. **try14** — leakage audit, prompted by TxGemma-predict voiding its own pre-try09
    verdicts (including the Gaucher pass this track's "complementarity" framing relied
    on). No hard leak: message passing uses train only, and every held-out node is
    blind. But **Fabry is blind only by luck** at the parent-category level
    (migalastat → *lysosomal lipid storage disorder* landed in the random validation
    draw), and **Gaucher has a real 2-hop sibling path in training** (miglustat →
    Niemann-Pick, which is legitimate knowledge but probably drives the rank of 17).
    `attention=True` was chosen after seeing Fabry's first result, so Fabry's 17 is
    optimistic. CF/TTR (try13) are the only fully prospective tests, and both were weak.
15. **try15** — Gaucher re-run with Niemann-Pick also held out (removes exactly one
    training edge: miglustat → Niemann-Pick). Tests whether miglustat's rank of 17
    depends on that sibling indication. Job 23210548, submitted 26-10-2026.
16. **try16** — connectivity vs rank across 120 diseases sampled across 6 log-degree
    bins and held out together (only for this test). Separates "disease node is well
    connected" from "a true drug already treats a graph-close disease". Job 23210549,
    submitted 26-10-2026; analyze with `analyze.py`.
17. **try17** — specificity check (CPU). **The model ranks by category, not by
    disease.** Fabry's and Gaucher's full drug rankings correlate 0.957 (41/50 shared
    top-50); CF's and TTR's 0.926; the two pairs share 0/50. Fabry's list ranks every
    rare-disease drug high: Gaucher drugs at 15–60, tafamidis at 46, CF modulators at
    61–69 (~20× better than for CF itself). Migalastat is 16 for Fabry vs 27 for
    Gaucher. So Fabry/Gaucher are category-level successes, not disease-specific;
    CF/TTR land in a steroid/chemo ranking mode instead. try16's analysis now includes
    a specificity metric.
18. **try18** — prototype-neighbour audit. **Fabry's result is leaked.** With
    proto=True a held-out disease is scored as 90% the blend of its 3 most similar
    training diseases. Fabry's #1 (31%) is *chylomicron retention disease*, which in
    PrimeKG carries all 23 of Fabry's genes (incl. GLA) **and a migalastat
    indication**. That's a data error (likely an "Anderson disease" name collision)
    that leaves Fabry's answer in training under another name. Gaucher's #1 (36%) is
    Niemann-Pick (miglustat). CF's are bronchiectasis (no CFTR modulator). TTR's are
    near-random (cos ≈ 0.03). The prototype neighbours' indications explain most of
    the Fabry/Gaucher vs CF/TTR gap. try16 analysis now records `proto_hit`. Fabry
    needs a re-run holding out the duplicate.
19. **try19** — option 1: per-drug popularity correction (z-score each drug against
    the same model's scores over 150 background diseases). The inference-only job
    23231183 reloads thesis checkpoints and verifies they reproduce saved scores
    (node input vectors are random/frozen and not checkpointed, so re-seeding
    matters).
20. **try20** — option 2: prototype reliability flag (best prototype cosine vs the
    paper's 0.2). TTR is unreliable under either signature (ATTRV122I 0.035,
    hereditary 0.000). The richer signature helps Fabry (demotes the leaky
    duplicate) but hurts CF.
21. **try21** — option 3: richer similarity signature (`all_nodes_profile_more`) on
    try16's exact 120-disease split, a paired comparison. Job 23231153. Variants are
    chosen on the sweep only, never on thesis diseases.
22. **try22** — option 4: learned blending gate, completing a 2×2 with the similarity
    signature on try16's split (jobs 23231350 `learn`, 23231351
    `learn_profile_more`). **Library bug**: `agg_measure='learn'` never learned,
    because `W_gate` is a plain dict (not registered, not trained, not saved).
    Patched `model.py` to wrap it in `nn.ModuleDict` only for 'learn' (other configs
    unchanged); scripts assert the patch is active.
