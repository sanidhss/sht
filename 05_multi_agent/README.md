# 05_multi_agent — tool-augmented multi-agent integrator

The fourth arm. Unlike 01/02/03, this one is **not** a disjoint evidence source — it is
an **integrator** that is allowed to call the other arms' evidence as tools. That
framing is a deliberate decision (see below), not an accident of design.

## The question

The first three arms test whether a single architecture, given one kind of evidence,
can nominate the right drug. This arm tests a different thing:

> When a general reasoning model is allowed to *orchestrate* — to call a property
> calculator, a binding-affinity scorer, and a knowledge graph, and to argue with
> itself about the result — does it beat the arms it is built out of, and does it beat
> the deterministic RDKit baseline that currently beats all of them?

Method: the RareAgent design (arXiv:2510.05764v1, Qin et al., Oct 2025) — an
adversarial Proponent/Skeptic debate over a per-query evidence graph, refereed by a PI
agent, with prompt self-evolution and a reusable heuristic library — adapted from its
original KG-only setting to this project's tool-augmented one.

## Why this exists: the gap it is meant to close

`02_txgemma_chat`'s README ends on an open question, and this project is the answer to
it. Restated:

> "Bare TxGemma" (a single `generate()` call, no retrieval) is not the same system as
> Google's benchmarked "Agentic-Tx" (Gemini 2.0 Pro orchestrating 18 tools, of which
> bare TxGemma is only one). The published rare-disease benchmark results are
> Agentic-Tx's, not bare TxGemma's. This project has only ever tested the
> no-retrieval configuration.

So every number this thesis reports for "the LLM arm" is a number for a configuration
Google never claimed would work. That is a real confound in the thesis's central
comparison, and closing it is this project's job.

## The bar

Explicit and already set by the other tracks. This arm has to beat:

1. **The RDKit descriptor baseline** — `library/02_txgemma_chat/try11/descriptor_baseline.py`
   (rank-within-filtered-set) and `library/01_txgemma_predict/.../descriptor_baseline_predict.py`
   (Tanimoto k-NN enrichment percentile). The chat arm loses to it on **4 of 4** folds
   by 2x–48x; the 9B fine-tune loses on **5 of 6** targets.
2. **Bare TxGemma-27b-chat**, same candidate pool — rank of the true drug out of 3,420.
   Current: Fabry 4, CF 993, Gaucher 1,143, TTR 1,774 (that last is exactly chance).

An integrator that calls RDKit cannot in principle lose to RDKit. If it does, that is a
finding about orchestration overhead, and it must be reported as one.

## Decisions made, and why (don't re-litigate without new evidence)

- **This is an integrator (4th arm), not a disjoint 3rd arm.** User decision,
  12-11-2026. The alternative — text/literature reasoning only, no chemistry or graph
  tools — would preserve the input-disjointness that makes the top-level README's
  complementarity claim meaningful, but would give up the one capability that fixes the
  measured failure mode ("the model is approximating a property calculator, more
  expensively and less accurately"). The complementarity result already stands on its
  own from 01 and 03; this arm tests something new instead of re-running 02 with more
  steps. **Consequence: this arm is compared *against* the other three, never counted
  as independent agreement with them.**

- **Deviation from the paper: the Explorer does not seed from a KG embedding.**
  RareAgent's Explorer scores candidates with a ComplEx-style bilinear function over
  PrimeKG, which is approximately what TxGNN already does. Seeding from it would make
  this arm's headline number partly a restatement of 03's. Candidate seeding comes from
  the existing approved-drug pool instead (`chembl_approved.json`, 3,417 compounds, the
  same pool 02 screens), so the ranking is comparable to 02's by construction. PrimeKG
  remains available as a *tool the agents may call*, which is a different thing from
  being the source of the candidate list.

- **Base model is chosen by measurement, not preference.** TxGemma-27b-chat has an
  8,192-token context with a 4,096 sliding window (verified in `config.json`:
  `model_type: gemma2`, `max_position_embeddings: 8192`, `sliding_window: 4096`). A
  debate protocol accumulates state per round, so the context budget is the binding
  constraint on whether the domain model can orchestrate at all. try01 measures this
  before any model is committed to. Precedent for taking it seriously: try05 in
  `02_txgemma_chat` was invalidated outright by a stale `max_new_tokens=500` silently
  truncating every completion — a token-budget bug that looked like a scientific result.

- **TxGemma stays in the design regardless of the orchestrator choice**, as a *tool*
  (`txgemma-9b-predict`, the 01 arm). Tool prompts are stateless and bounded — the
  stock TDC template is 415–977 tokens for this project's targets, ~3,300 for RYR1 at
  5,038 aa — so the 8K limit never binds on them. This mirrors Agentic-Tx, where
  TxGemma is a tool and a general model orchestrates.

- **Constrained decoding, not prompt-begging, for the JSON protocol.** RareAgent
  requires every agent turn to emit a single strict JSON object. Guided decoding makes
  malformed output impossible rather than unlikely, which removes a mechanical failure
  mode that would otherwise be indistinguishable from "multi-agent doesn't help."

## Blinding — the unsolved problem

**This arm cannot be blinded the way the others are.** 01 holds out training compounds;
03 holds out graph edges. This arm does no training at all, so its knowledge lives in
pretrained weights that cannot be edited. Every drug in `pins.json` is well documented
in the literature the base model was trained on.

`01_txgemma_predict`'s try13 already measured this effect and found that **success
tracks the base model's prior, not the training data**. Lining its familiarity
percentile up against 02's screen rank:

| disease | drug | familiarity %ile (9B) | 02 chat rank (27B) |
|---|---|---|---|
| Fabry | migalastat | 0.2 | 4 |
| Gaucher | ambroxol | 5.4 | 1,143 |
| CF | lumacaftor | 42.4 | 993 |
| ATTR | tafamidis | 70.2 | 1,774 |

Near-monotonic. The one spectacular result in 02 (Fabry, 427x enrichment) is also the
most-memorized molecule in the project. **Caveat: these are two different models (9B
predict vs 27B chat), so this is a hypothesis, not an established correlation** — but it
is cheap to test directly and must be tested before any number here is written up.

A larger base model makes this strictly worse. Planned mitigations, in order of cost:

1. **Familiarity probe on the chosen base model** (try13's method, ~1 GPU-hour). Report
   every result against it. Non-negotiable prerequisite.
2. **Blinded disease description** — gene, mechanism and phenotype only, never the
   disease name. `coverage_audit.csv` already carries `gene`/`uniprot`/`site`/`mech`.
3. **Date-cut retrieval corpus** — restrict agent evidence to literature published
   before the reference drug's approval. Most faithful, by far the most work; its own
   phase, not part of the first build.

## Status

- **try01 (13-11-2026) — context budget audit. DONE.** The protocol does not fit
  `txgemma-27b-chat`: the PI role is at 183% of the 8,192-token limit in round 1 and
  692% by round 4, and the overrun is in the accumulating role, not the shrinkable one.
  TxGemma could orchestrate only k≤3 candidates over 2 rounds, which is a different and
  much weaker method, not the same method scaled down. **Orchestration goes to
  Llama-3.3-70B; TxGemma stays as a bounded tool.** Also found: k=50 over 4 rounds
  overruns Llama's 131,072 too (138,709), so k is capped at ~20–30 for a 4-round run.
- Next: vLLM environment, Llama-3.3-70B weights (**gated — needs a license acceptance
  on the user's HF account**), familiarity probe, then the runner.

### The 2×2, as try01 leaves it

|  | single-shot | multi-agent |
|---|---|---|
| **TxGemma-27b-chat** | done — `02_txgemma_chat` (Fabry 4, CF 993, Gaucher 1,143, TTR 1,774) | **not runnable at protocol scale** — try01, recorded reason |
| **Llama-3.3-70B** | to run (controls for base-model capability) | to run (the actual proposal) |

The bottom row is what separates "multi-agent scaffolding helps" from "a stronger base
model helps" — without it, the two are confounded and neither can be claimed.

## Where everything is

- Code + try history (git, pushed): this directory.
- Session home (local-only, not in git): `/projects/bisa/sshah21/multi_LLM/`
- Candidate pool: `/work/hdd/bisa/sshah21/results/reasoning_screen/chembl_approved.json`
  (3,417 approved drugs; 02 spikes 3 authored analogs to reach 3,420)
- Disease/drug pins: `/projects/bisa/sshah21/pins.json` (12 cells)
- Target coverage + structure gates: `/projects/bisa/sshah21/coverage_audit.csv`
- Results (not in git): `/work/hdd/bisa/sshah21/results/multi_agent/`
