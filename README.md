# Library index

**For a coding assistant:** if you're picking this project up in a new session, read
[AGENT_INSTRUCTIONS.txt](AGENT_INSTRUCTIONS.txt) first — it defines what you're
expected to write into trial READMEs (and when) as part of this project's workflow.

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
  fine-tune (`txgemma-9b-predict`). 5 tries. Current state: Gaucher passes a real
  hard-negative test, Fabry doesn't yet.
- **[02_txgemma_chat](02_txgemma_chat/README.md)** — reasoning-based chaperone
  identification (`txgemma-27b-chat`), culminating in a leave-one-target-out fine-tune
  across 4 disease mechanisms. 4 tries. Current state: no successful training run yet,
  latest bug just fixed.
- **[03_txgnn](03_txgnn/README.md)** — PrimeKG knowledge-graph link prediction, no
  chemistry involved. 9 tries, the longest debugging arc. Current state: infrastructure
  and library bugs fully resolved as of try09; not yet run under the final fix.
- **[04_setup](04_setup/README.md)** — one-time environment/data setup (model weights,
  PrimeKG). 2 tries, both resolved cleanly.

## Cross-project context
All three model tracks (01/02/03) are independent attempts at the same underlying
question: can a computational tool correctly identify pharmacological
chaperones/correctors/stabilizers for orphan disease targets (Fabry, Gaucher, and later
cystic fibrosis and TTR amyloidosis) with genuine, verifiable blindness to the true
answer during training/evaluation — not just easy discrimination against obviously
unrelated compounds. See each project's README for its own arc; the "hard negative"
discipline (real structurally-similar analogs, not just unrelated drugs) recurs across
all three as the thing that separates a real result from an easy, uninformative one.
