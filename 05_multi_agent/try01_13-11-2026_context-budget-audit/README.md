# Try 01 — context budget audit

**Date:** 13-11-2026
**SLURM job:** interactive `srun` on `cpu` (jobs 23850401, 23850410) — CPU only, no GPU,
no model weights loaded
**Scripts (exact versions run, snapshotted at submission time):**
- `context_audit.py`
- `protocol.py`

## What it does

Settles, by measurement rather than preference, which base model can **orchestrate** the
multi-agent protocol.

`txgemma-27b-chat` has an 8,192-token context with a 4,096 sliding window (verified
directly in its `config.json` on disk: `model_type: gemma2`,
`max_position_embeddings: 8192`, `sliding_window: 4096`). A debate protocol accumulates
state every round — evidence graph, transcript, tool results — so context is the binding
constraint on whether the domain-tuned model can hold the orchestrator role at all.

The audit imports the **real** prompt templates from `protocol.py` (so it cannot drift
from what is actually sent), fills them with real target data from `coverage_audit.csv`
and real SMILES from the 3,417-compound approved library, grows the evidence graph and
transcript the way the protocol grows them, and tokenises with the real Gemma
tokeniser. Free-text filler is generated at exactly the word caps the JSON schemas
permit, so the numbers are a worst case the schema actually allows rather than an
optimistic estimate from short toy strings.

This was run first, before any model was committed to, because of direct precedent:
`02_txgemma_chat` try05 was invalidated outright by a stale `max_new_tokens=500`
silently truncating every completion. A protocol that overruns its context fails the
same way — quietly, and looking like a scientific result.

## What happened

**The protocol does not fit `txgemma-27b-chat`, and the overrun is structural.**

Default settings (k=20 candidates, 4 rounds, 3 nodes + 2 edges per agent per round,
J=5 heuristics, target = mastocytosis/KIT):

| round | explorer | proponent | skeptic | **PI** |
|---|---|---|---|---|
| 1 | 8,455 | 2,267 | 2,319 | **14,999** |
| 2 | 8,455 | 3,199 | 3,251 | **28,699** |
| 3 | 8,455 | 4,145 | 4,197 | **42,679** |
| 4 | 8,455 | 5,094 | 5,146 | **56,719** |

The PI is the blocker, and it is over budget **at round 1** — 183% of the limit before
the debate has accumulated anything. The cause is structural: the PI referees all
candidates at once, so its prompt carries every candidate's evidence graph, and it grows
in both k and rounds. By round 4 it is 692% of the limit.

The Proponent and Skeptic fit comfortably (max 5,146), because each sees only the one
candidate under debate. The Explorer's 8,455 is marginal but **not** interesting — it is
the one term that shrinks trivially by sending fewer library compounds per call.

So the overrun sits in exactly the role that cannot be trimmed without changing the
method.

### Sensitivity sweep — what would fit

| k | rounds | worst debate/PI prompt | fits 8,192? | fits 131,072? |
|---|---|---|---|---|
| 3 | 2 | 5,613 | **yes** | yes |
| 5 | 2 | 8,329 | no (just over) | yes |
| 5 | 3 | 12,019 | no | yes |
| 10 | 3 | 22,239 | no | yes |
| 20 | 4 | 56,719 | no | yes (43%) |
| 50 | 4 | 138,709 | no | **no** |

Two findings here, not one:

1. **TxGemma could orchestrate only k≤3 candidates over 2 rounds.** That is not the
   method with a smaller parameter — it is a different and much weaker method.
   RareAgent's result depends on adversarial debate accumulating evidence across rounds;
   two rounds over three candidates barely exercises it. Reporting that as "multi-agent
   doesn't help" would be a measurement artefact of the context window, which is
   precisely the try05 failure mode repeated.
2. **k=50 over 4 rounds overruns Llama-3.3-70B as well** (138,709 vs 131,072). The
   long-context model is not unlimited, and the PI's all-candidates prompt is what
   consumes it. **Design constraint going forward: cap k at ~20–30 for a 4-round run**,
   or shard the PI's scoring across candidate batches. Worth knowing now rather than
   discovering it as a truncation bug mid-run.

### Verdict

Orchestration goes to the long-context model. `txgemma-9b-predict` stays in the design
as a **tool**, where the limit never binds: the stock TDC prompt is one SMILES plus one
protein sequence, 415–977 tokens for this project's targets and ~3,300 for RYR1 at
5,038 aa, stateless and non-accumulating.

This independently reproduces Google's own architectural choice in Agentic-Tx — a
general model orchestrates, TxGemma is one of the tools — and it is now justified by a
measurement from this project rather than by deference to their design.

**Consequence for the 2×2:** the "TxGemma × multi-agent" cell cannot be run at the full
protocol. The honest 2×2 is therefore Llama-3.3-70B × {single-shot, multi-agent} plus
TxGemma-27b-chat single-shot (which `02_txgemma_chat` already has), with the fourth cell
reported as *not runnable at protocol scale, for a recorded reason* rather than left
blank or faked at k=3.

## Infrastructure notes (cost real time; recorded so they aren't rediscovered)

- **`import transformers` takes 197 s** in `envs/txgemma` on `/work/hdd` (Lustre), before
  any tokeniser is touched. Two 10-minute interactive allocations died to this. The
  audit now loads `tokenizer.json` through `tokenizers.Tokenizer.from_file` instead —
  same vocabulary, same ids, ~1 s import. **Any short CPU job in this env should avoid
  importing `transformers`.**
- **The assistant scratchpad (`/tmp/...`) is login-node-local and invisible to compute
  nodes.** Anything a compute job must read has to live on `/projects` or `/work`.
- Compute nodes have no outbound HTTPS, so `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  should be set for anything touching the HF cache. (Not the cause of the hang above,
  but it would have been the next one.)

## Caveats

- Token counts use the **Gemma** tokeniser (256k vocab). Llama-3.3 uses a different
  128k-vocab tokeniser and will differ by a few percent. That is irrelevant to the
  verdict — the TxGemma overrun is 183–692%, far outside tokeniser variance — but the
  k=50 vs Llama finding (138,709 vs 131,072, only 6% over) **should be re-measured with
  the real Llama tokeniser** before being treated as a hard limit.
- `NODES_PER_TURN=3` / `EDGES_PER_TURN=2` are this implementation's values; the paper
  does not report its own. The conclusion is conditional on them, though the PI's
  round-1 overrun (183%) holds even at one node per turn.
- The free-text filler is at the schema's word caps. Real agent output will often be
  shorter, so typical prompts will be smaller than these — but a protocol must be sized
  for what its schema permits, not for its average case.
