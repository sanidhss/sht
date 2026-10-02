# Library index

Organized by project, then by try (one folder per distinct attempt, named
`tryNN_DD-MM-YYYY_short-description`). Each try folder contains the scripts as they
existed at that point (where separately preserved — several were edited in place across
iterations and only the final version survives; noted explicitly where that's the case),
the real SLURM job log(s), and a `README.md` explaining what it does and, if it failed,
why.

Model weights, PrimeKG's database, HuggingFace cache, and other large data artifacts are
**not** duplicated here — they remain at their original paths under
`/work/hdd/bisa/sshah21/` (`hf_cache/`, `primekg/`, `results/`). This library holds code
and logs only.

## Projects

- **[01_txgemma_predict](01_txgemma_predict/README.md)** — numeric drug-target scoring
  fine-tune (`txgemma-9b-predict`), chemistry only (SMILES + real bioactivity data).
  6 tries. Current state: Gaucher passes a real hard-negative test; Fabry fails it
  (traced to a 13.5x real-data gap vs. Gaucher, not fixable by more augmentation);
  CF/TTR running under the existing general pipeline as the real test of whether this
  approach extrapolates beyond lysosomal enzymes. See
  `NEXT_PHASE_diverse_structural_training.md` for the planned next phase.
- **[02_txgemma_chat](02_txgemma_chat/README.md)** — reasoning-based chaperone
  identification (`txgemma-27b-chat`), leave-one-target-out across 4 mechanism
  classes. 7 tries. Current state: compositional per-candidate verdict redesign shows
  real partial transfer (logic-consistency well above chance, 3/4 folds land in the
  right structural family cold) but blends/retrieves a training target's condition
  when two mechanisms share surface vocabulary; a diagnostic to isolate that failure
  mode is in flight.
- **[03_txgnn](03_txgnn/README.md)** — PrimeKG knowledge-graph link prediction, no
  chemistry at all. 11 tries, the longest debugging arc. Current state: Fabry gets a
  strong, consistent top-0.2% result; Gaucher is noisy per-seed but reaches a
  comparable result once ensembled across seeds (the variance turned out to be
  inherent embedding stochasticity, not a fixable training bug — tested and
  confirmed, not assumed).
- **[04_setup](04_setup/README.md)** — one-time environment/data setup (model weights,
  PrimeKG). 2 tries, both resolved cleanly.

## Cross-project context
All three model tracks (01/02/03) are independent attempts at the same underlying
question: can a computational tool correctly identify pharmacological
chaperones/correctors/stabilizers for orphan disease targets (Fabry, Gaucher, and later
cystic fibrosis and TTR amyloidosis) with genuine, verifiable blindness to the true
answer during training/evaluation — not just easy discrimination against obviously
unrelated compounds. The "hard negative" discipline (real structurally-similar
analogs, not just unrelated drugs) recurs across all three as the thing that separates
a real result from an easy, uninformative one.

**The sharpest finding so far spans two of the three tracks**: TxGemma-predict
(chemistry only) passes on Gaucher and fails on Fabry; TxGNN (pure graph topology,
zero chemistry) does the opposite — strong on Fabry, weaker on Gaucher. These two
model families draw on completely disjoint evidence (one never sees a SMILES string,
the other never sees bioactivity data), and they succeed on opposite diseases. That's
evidence the two signal sources are genuinely complementary, not redundant — which
disease is "easy" depends on which evidence source happens to be rich for it, not on
any property of the disease alone. Deliberately decided *not* to merge chemistry into
TxGNN's architecture (would blur this clean contrast and risks a much larger,
riskier library modification) — the right next step, once both tracks look
individually promising on more than 2 diseases, is embedding/ensembling the two
models' outputs, not merging their internals.

**Also deliberately decided**: not to keep deepening Fabry-specific fixes (e.g.
stereochemistry/diastereomer-specific training) in isolation — doing so risks
teaching a narrow heuristic that happens to work for lysosomal enzymes with no
evidence it extrapolates to a disease with a genuinely different receptor profile,
which is the actual thing this thesis needs. CF/TTR running under the *unmodified*
general pipeline (01) is the real test of extrapolation breadth.

## For a fresh Claude Code session
Each project has its own dedicated, local-only (not pushed to GitHub) Claude Code
session home with a `CLAUDE.md` briefing: `$PROJ/cc_01_txgemma_predict`,
`cc_02_txgemma_chat`, `cc_03_txgnn`, `cc_04_setup`. Start there for that project's
context; this file plus each project's own README remain the authoritative, versioned
record of what actually happened.
