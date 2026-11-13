"""
Context-budget audit for the multi-agent protocol (try01).

Question it answers: does a RareAgent-style debate protocol fit inside
txgemma-27b-chat's 8,192-token context (4,096 sliding window), or does orchestration
require a long-context model?

This is measured, not estimated. It imports the real prompt templates from
`protocol.py`, fills them with real target data from `coverage_audit.csv` and real
SMILES from the approved-drug library, grows the evidence graph and debate transcript
the way the protocol actually grows them, and tokenises every resulting prompt with the
real tokeniser.

Why it matters enough to run first: `02_txgemma_chat` try05 was invalidated outright by
a stale `max_new_tokens=500` silently truncating every completion. A protocol that
overruns its context fails the same way — quietly, and looking like a scientific result.

CPU only, no GPU, no model weights. Needs `transformers` (for the tokeniser) and the
tokeniser files already in the HF cache.

Usage:
    python context_audit.py                      # default sweep
    N_CANDIDATES=10 MAX_ROUNDS=3 python context_audit.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol as P  # noqa: E402

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------

PROJ = Path("/projects/bisa/sshah21")
WORK = Path("/work/hdd/bisa/sshah21")

COVERAGE_CSV = PROJ / "coverage_audit.csv"
LIBRARY_JSON = WORK / "results/reasoning_screen/chembl_approved.json"

# txgemma-27b-chat: verified from its config.json on disk.
GEMMA_LIMIT = 8192
GEMMA_SLIDING = 4096
# Llama-3.3-70B-Instruct published context.
LLAMA_LIMIT = 131072

TOKENISER_PATHS = {
    "txgemma-27b-chat (gemma2)": (
        WORK / "hf_cache/hub/models--google--txgemma-27b-chat/snapshots"
    ),
}

N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "20"))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "4"))
N_HEURISTICS = int(os.environ.get("N_HEURISTICS", "5"))
# Per agent, per round, per candidate. RareAgent does not report these; these are the
# values this implementation uses, and the audit's conclusion is conditional on them.
NODES_PER_TURN = 3
EDGES_PER_TURN = 2

TARGET_KEY = os.environ.get("TARGET", "mastocytosis")


# --------------------------------------------------------------------------------
# Realistic filler
# --------------------------------------------------------------------------------
# The protocol caps every free-text field by word count. Filler is generated at exactly
# those caps, so the measurement is a worst case the schema actually permits — not a
# guess, and not an underestimate from short toy strings.

_LOREM = (
    "the compound presents a rigid fused bicyclic core whose hydrogen bond donor "
    "geometry is compatible with the reported binding pocket and shows measured "
    "affinity in a comparable human binding assay against this protein family which "
    "supports a direct interaction rather than an indirect downstream effect observed "
    "in cellular readouts across several independent published experimental reports"
).split()


def words(n: int) -> str:
    out = []
    while len(out) < n:
        out.extend(_LOREM)
    return " ".join(out[:n])


def make_graph(candidate: str, n_nodes: int, n_edges: int) -> P.EvidenceGraph:
    g = P.EvidenceGraph(candidate=candidate)
    for i in range(n_nodes):
        g.nodes.append(
            P.EvidenceNode(
                id=f"{candidate}-n{i}",
                claim=words(40),  # schema cap
                source=f"tool:tanimoto_neighbours#{i}",
                confidence=1.0 - i * 0.01,
            )
        )
    for i in range(n_edges):
        g.edges.append(
            P.EvidenceEdge(
                src=f"{candidate}-n{i}",
                dst=f"{candidate}-n{(i + 1) % max(n_nodes, 1)}",
                kind="support" if i % 3 else "refute",
                rationale=words(25),  # schema cap
            )
        )
    return g


# --------------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------------


def load_target(key: str) -> dict[str, str]:
    with COVERAGE_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if key.lower() in r["disease"].lower():
            return r
    raise SystemExit(
        f"target {key!r} not found in {COVERAGE_CSV}; available: "
        + ", ".join(sorted({r['disease'] for r in rows}))
    )


def load_library(n: int) -> list[dict[str, str]]:
    if not LIBRARY_JSON.exists():
        # Fall back to representative drug-like SMILES lengths so the audit still runs.
        smi = "CC(=O)Nc1ccc(O)cc1" * 3
        return [{"id": f"L{i:04d}", "smiles": smi} for i in range(n)]
    data = json.loads(LIBRARY_JSON.read_text())
    recs = data if isinstance(data, list) else list(data.values())
    out = []
    for i, r in enumerate(recs[:n]):
        out.append(
            {
                "id": f"L{i:04d}",
                "smiles": r.get("canonical_smiles") or r.get("smiles") or "",
            }
        )
    return out


def find_tokeniser():
    """Load the tokeniser via `tokenizers`, NOT `transformers`.

    `import transformers` measured at 197s on this Lustre filesystem (conda env over
    /work/hdd), which alone overran a 10-minute interactive allocation before any
    tokenising happened. `tokenizers.Tokenizer.from_file` reads the single tokenizer.json
    and imports in about a second. Same vocabulary, same merges, same token ids.
    """
    from tokenizers import Tokenizer

    for label, base in TOKENISER_PATHS.items():
        if not base.exists():
            continue
        for snap in sorted(p for p in base.iterdir() if p.is_dir()):
            tj = snap / "tokenizer.json"
            if not tj.exists():
                continue
            try:
                return label, Tokenizer.from_file(str(tj))
            except Exception as exc:  # pragma: no cover
                print(f"  (could not load {label}: {exc})", file=sys.stderr)
    raise SystemExit("no usable tokenizer.json found in the HF cache")


def n_tokens(tok, text: str) -> int:
    """Token count, for either a `tokenizers.Tokenizer` or a transformers tokeniser."""
    if hasattr(tok, "encode_batch"):  # tokenizers.Tokenizer
        return len(tok.encode(text, add_special_tokens=False).ids)
    return len(tok(text, add_special_tokens=False)["input_ids"])


# --------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------


def build_prompts(target: dict[str, str], library: list[dict[str, str]], rnd: int):
    """Every prompt the protocol sends at round `rnd` (1-indexed)."""
    tgt = P.blinded_target_block(
        gene=target["gene"],
        uniprot=target["uniprot"],
        sequence_len=400,
        site=target["site"],
    )
    mech = P.mechanism_block(
        mech=target["mech"],
        phenotypes=[words(3) for _ in range(8)],
        free_text=words(120),
    )
    tools = P.tool_block()
    cutoff = "2011-01-01"

    # Graph and transcript both grow with the round number.
    n_nodes = NODES_PER_TURN * 2 * rnd
    n_edges = EDGES_PER_TURN * 2 * rnd
    graphs = [make_graph(c["id"], n_nodes, n_edges) for c in library[:N_CANDIDATES]]
    transcript = [
        {"role": r, "argument": words(120)}
        for _ in range(rnd)
        for r in ("proponent", "skeptic")
    ]
    heuristics = [{"when": words(25), "then": words(25)} for _ in range(N_HEURISTICS)]

    cand_block = f"{library[0]['id']}\t{library[0]['smiles']}"
    one_graph = graphs[0].serialise()
    all_graphs = "\n".join(g.serialise() for g in graphs)
    tr = P.transcript_block(transcript)

    out = {}
    out["explorer"] = P.build(
        "explorer",
        k=N_CANDIDATES,
        target_block=tgt,
        mechanism_block=mech,
        cutoff_date=cutoff,
        library_block=P.library_block(library),
    )
    for role in ("proponent", "skeptic"):
        out[role] = P.build(
            role,
            target_block=tgt,
            mechanism_block=mech,
            cutoff_date=cutoff,
            candidate_block=cand_block,
            graph_block=one_graph,
            transcript_block=tr,
            tool_block=tools,
        )
    out["pi"] = P.build(
        "pi",
        target_block=tgt,
        mechanism_block=mech,
        cutoff_date=cutoff,
        heuristics_block=P.heuristics_block(heuristics),
        graphs_block=all_graphs,
        transcript_block=tr,
    )
    return out


def main() -> None:
    target = load_target(TARGET_KEY)
    label, tok = find_tokeniser()
    library = load_library(max(N_CANDIDATES, 200))

    print("=" * 78)
    print("CONTEXT BUDGET AUDIT — multi-agent protocol")
    print("=" * 78)
    print(f"tokeniser          : {label}")
    print(f"target             : {target['disease']} ({target['gene']}/{target['uniprot']})")
    print(f"candidates (k)     : {N_CANDIDATES}")
    print(f"rounds             : {MAX_ROUNDS}")
    print(f"nodes/turn/agent   : {NODES_PER_TURN}   edges/turn/agent: {EDGES_PER_TURN}")
    print(f"heuristics (J)     : {N_HEURISTICS}")
    print(f"library in Explorer: {len(library)} compounds")
    print()
    print(f"txgemma-27b-chat limit : {GEMMA_LIMIT} (sliding window {GEMMA_SLIDING})")
    print(f"llama-3.3-70b limit    : {LLAMA_LIMIT}")
    print()

    rows = []
    print(f"{'round':>5}  {'role':<10} {'tokens':>8}  {'vs 8192':>9}  verdict")
    print("-" * 78)
    worst = 0
    for rnd in range(1, MAX_ROUNDS + 1):
        prompts = build_prompts(target, library, rnd)
        for role, text in prompts.items():
            n = n_tokens(tok, text)
            worst = max(worst, n)
            pct = 100.0 * n / GEMMA_LIMIT
            verdict = "OK" if n <= GEMMA_LIMIT else "OVERRUN"
            if n > GEMMA_SLIDING and n <= GEMMA_LIMIT:
                verdict = "OK (beyond sliding window)"
            print(f"{rnd:>5}  {role:<10} {n:>8}  {pct:>8.1f}%  {verdict}")
            rows.append({"round": rnd, "role": role, "tokens": n})
        print("-" * 78)

    print()
    print(f"worst-case prompt: {worst} tokens  ({100.0 * worst / GEMMA_LIMIT:.1f}% of 8192, "
          f"{100.0 * worst / LLAMA_LIMIT:.2f}% of 131072)")
    print()

    # The Explorer's library block is the one term that is trivially shrinkable, so
    # report it separately: an overrun driven only by the Explorer is fixable, an
    # overrun in the PI or the debate roles is structural.
    explorer_rows = [r for r in rows if r["role"] == "explorer"]
    debate_rows = [r for r in rows if r["role"] in ("proponent", "skeptic", "pi")]
    worst_debate = max(r["tokens"] for r in debate_rows)
    print(f"worst Explorer prompt  : {max(r['tokens'] for r in explorer_rows)} "
          f"(shrinkable: send fewer library compounds per call)")
    print(f"worst debate/PI prompt : {worst_debate} "
          f"(structural: grows with rounds and candidates)")
    print()

    if worst_debate > GEMMA_LIMIT:
        print("VERDICT: the debate protocol does NOT fit txgemma-27b-chat's context.")
        print("  The overrun is in the accumulating roles, not the shrinkable one, so it")
        print("  cannot be fixed by trimming the candidate library. Orchestration needs a")
        print("  long-context model; txgemma stays in the design as a bounded tool.")
    else:
        print("VERDICT: the debate protocol fits txgemma-27b-chat's context at these")
        print("  settings. The clean single-variable comparison (same model, with and")
        print("  without scaffolding) is available.")

    out_path = Path(os.environ.get("AUDIT_OUT", "context_audit_results.json"))
    out_path.write_text(
        json.dumps(
            {
                "tokeniser": label,
                "target": target["disease"],
                "settings": {
                    "n_candidates": N_CANDIDATES,
                    "max_rounds": MAX_ROUNDS,
                    "nodes_per_turn": NODES_PER_TURN,
                    "edges_per_turn": EDGES_PER_TURN,
                    "n_heuristics": N_HEURISTICS,
                    "library_in_explorer": len(library),
                },
                "limits": {"gemma": GEMMA_LIMIT, "gemma_sliding": GEMMA_SLIDING,
                           "llama": LLAMA_LIMIT},
                "rows": rows,
                "worst_overall": worst,
                "worst_debate": worst_debate,
                "fits_gemma": worst_debate <= GEMMA_LIMIT,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
