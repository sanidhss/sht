"""
Agent protocol for the tool-augmented multi-agent integrator (05_multi_agent).

This module is the single source of truth for what every agent turn actually looks
like: role prompts, the strict-JSON output schemas, the evidence-graph serialisation,
and the tool catalogue. `context_audit.py` imports it to measure the token budget, and
the runner imports the same objects to execute — so the audit can never drift from
what is really sent to the model.

Design follows RareAgent (arXiv:2510.05764v1) with two recorded deviations, both in
`README.md`:
  1. The Explorer does not seed from a KG embedding (that would restate the 03 arm).
  2. Agents may call tools, because this arm is an integrator, not a disjoint source.

Nothing here imports torch or rdkit at module level, so the audit runs on a bare CPU
node with only `transformers` available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------------
# Tool catalogue
# --------------------------------------------------------------------------------
# Every tool is stateless and bounded: one call, one JSON result, no history. That is
# what lets `txgemma-9b-predict` participate even though its 8K context cannot hold a
# debate (README, "TxGemma stays in the design").

TOOLS: dict[str, dict[str, Any]] = {
    "rdkit_descriptors": {
        "description": (
            "Compute physicochemical descriptors for a SMILES string. Deterministic, "
            "exact, and always available. Returns MW, cLogP, TPSA, HBD, HBA, "
            "rotatable bonds, aromatic rings, formal charge, QED and the Bemis-Murcko "
            "scaffold."
        ),
        "args": {"smiles": "str"},
        "returns": (
            "{mw, clogp, tpsa, hbd, hba, rotb, arom_rings, formal_charge, qed, "
            "scaffold_smiles}"
        ),
    },
    "txgemma_score": {
        "description": (
            "Predicted binding affinity for a (compound, protein target) pair from "
            "txgemma-9b-predict, on a per-target normalised log-IC50 scale from 000 to "
            "1000 where LOWER means MORE POTENT. The scale is per-target and is never "
            "comparable across targets. Treat it as one noisy opinion, not ground truth."
        ),
        "args": {"smiles": "str", "uniprot": "str"},
        "returns": "{score: float, scale: 'per_target_norm_log_ic50', lower_is_better: true}",
    },
    "tanimoto_neighbours": {
        "description": (
            "The k structurally most similar compounds to a SMILES among compounds with "
            "MEASURED activity against this target, with their real activity values. "
            "This is the strongest non-LLM signal available; use it before arguing from "
            "structure alone."
        ),
        "args": {"smiles": "str", "uniprot": "str", "k": "int"},
        "returns": "[{smiles, chembl_id, similarity, pchembl, assay_type}]",
    },
    "chembl_activity": {
        "description": (
            "Measured bioactivity records for a (compound, target) pair from ChEMBL, "
            "restricted to human SINGLE PROTEIN binding assays with a pchembl value. "
            "Empty result means no measurement exists, NOT that the compound is inactive."
        ),
        "args": {"smiles": "str", "uniprot": "str"},
        "returns": "[{pchembl, standard_type, standard_value, units, assay_chembl_id}]",
    },
    "primekg_neighbours": {
        "description": (
            "Graph neighbours of a node in PrimeKG, by relation type. Available for "
            "drugs, genes/proteins, diseases and phenotypes. NOTE: answer-bearing edges "
            "(indication, off-label use) for the disease under study have been removed "
            "from the graph this tool reads; a missing indication edge therefore carries "
            "no information."
        ),
        "args": {"node_name": "str", "relation": "str|null"},
        "returns": "[{relation, neighbour_name, neighbour_type}]",
    },
    "literature_search": {
        "description": (
            "Search the date-cut literature corpus. Returns title + abstract snippets. "
            "The corpus contains ONLY records published strictly before this case's "
            "evidence cutoff date, which is stated in your task prompt. Nothing "
            "published after that date is retrievable, by construction."
        ),
        "args": {"query": "str", "k": "int"},
        "returns": "[{pmid, year, title, snippet}]",
    },
}


# --------------------------------------------------------------------------------
# Evidence graph
# --------------------------------------------------------------------------------

EdgeKind = Literal["support", "refute", "entail"]


@dataclass
class EvidenceNode:
    """One claim in the task evidence graph (T-EGraph in the paper)."""

    id: str
    claim: str
    source: str  # "tool:rdkit_descriptors" | "agent:proponent" | "literature:PMID"
    confidence: float


@dataclass
class EvidenceEdge:
    src: str
    dst: str
    kind: EdgeKind
    rationale: str


@dataclass
class EvidenceGraph:
    candidate: str
    nodes: list[EvidenceNode] = field(default_factory=list)
    edges: list[EvidenceEdge] = field(default_factory=list)

    def serialise(self, max_nodes: int | None = None) -> str:
        """Compact JSON form that goes into a prompt.

        `max_nodes` keeps the highest-confidence nodes only. This is the single knob
        that decides whether the protocol fits an 8K context, so the audit sweeps it
        rather than guessing a value.
        """
        nodes = sorted(self.nodes, key=lambda n: -n.confidence)
        if max_nodes is not None:
            nodes = nodes[:max_nodes]
        keep = {n.id for n in nodes}
        return json.dumps(
            {
                "candidate": self.candidate,
                "nodes": [
                    {"id": n.id, "claim": n.claim, "src": n.source, "conf": round(n.confidence, 2)}
                    for n in nodes
                ],
                "edges": [
                    {"s": e.src, "d": e.dst, "k": e.kind, "why": e.rationale}
                    for e in self.edges
                    if e.src in keep and e.dst in keep
                ],
            },
            separators=(",", ":"),
        )


# --------------------------------------------------------------------------------
# Role prompts
# --------------------------------------------------------------------------------
# Written to be explicit about the two things that went wrong in 02_txgemma_chat:
#   - try09: the model invented a condition ("di-metal-binding hexose") and then let it
#     steer the whole screen. Hence the grounding requirement below.
#   - try11: a ~270x swing from the phrasing register of the condition alone. Hence the
#     requirement to cite a tool result or a retrieved record for every claim.

_COMMON_RULES = """\
Rules that apply to every response you give:
1. Output a single UTF-8 JSON object and nothing else. No Markdown, no prose, no code
   fences, no commentary before or after.
2. Every claim you assert must carry a `support` field naming either a tool result you
   actually received in this conversation, or a literature record id you actually
   retrieved. If you have neither, you must either call a tool or mark the claim
   `"speculative": true`. Asserting an unsupported claim as fact is the single most
   serious protocol violation.
3. You may not use the identity of any approved drug for the condition under study as
   evidence, even if you believe you know it. Reason from the mechanism, the structure
   and the measurements in front of you. If you catch yourself pattern-matching to a
   remembered drug-disease pair, say so in `recall_flag` and continue reasoning without
   it.
4. Physicochemical quantities (logP, MW, TPSA, counts) must come from
   `rdkit_descriptors`, never from your own estimation. You are worse at arithmetic over
   structures than the tool is, and the tool is free.
"""

EXPLORER_PROMPT = """\
You are the Explorer in a multi-agent drug-repurposing investigation.

You are given a protein target and a mechanistic description of a disease state. You
are NOT given the disease's name, and you must not try to infer it in order to recall
its approved drug.

Your job: propose {k} candidate compounds from the supplied library that are worth
investigating, and state for each the ONE structural or physicochemical property that
makes it worth a closer look. You are seeding an investigation, not deciding it — a
candidate you propose will be attacked by a Skeptic, so propose things whose case can
actually be made from evidence.

{common_rules}

Target:
{target_block}

Mechanism under study:
{mechanism_block}

Evidence cutoff: {cutoff_date}. Nothing published after this date is available to any
agent in this investigation.

Library (compound id, name withheld, SMILES):
{library_block}

Respond with this exact schema:
{{"candidates":[{{"id":"<library id>","why":"<one property, <=30 words>","support":"<tool result or record id>","speculative":<bool>}}],"recall_flag":"<empty string, or what you caught yourself recalling>"}}
"""

PROPONENT_PROMPT = """\
You are the Proponent. You argue FOR the candidate below being a genuine therapeutic
agent for the mechanism under study.

You are not a salesperson. A case that collapses under the Skeptic's first question
costs the investigation a round and will be scored against you by the PI. Argue only
what the evidence supports, and call tools to get evidence you do not have.

{common_rules}

Target:
{target_block}

Mechanism under study:
{mechanism_block}

Evidence cutoff: {cutoff_date}.

Candidate under debate:
{candidate_block}

Evidence graph so far:
{graph_block}

Debate so far:
{transcript_block}

Tools available:
{tool_block}

Respond with this exact schema:
{{"tool_calls":[{{"tool":"<name>","args":{{}}}}],"new_nodes":[{{"id":"<new id>","claim":"<=40 words","source":"<tool or record>","confidence":<0..1>}}],"new_edges":[{{"s":"<id>","d":"<id>","k":"support|refute|entail","why":"<=25 words"}}],"argument":"<=120 words","recall_flag":"<empty string, or what you caught yourself recalling>"}}
"""

SKEPTIC_PROMPT = """\
You are the Skeptic. You argue AGAINST the candidate below, and your job is to find the
reason it fails, not to be contrarian.

The strongest attacks are usually: a measured value that contradicts the Proponent's
structural argument; a property the compound cannot physically satisfy (the tool will
tell you); a close structural analog with measured inactivity; or an argument that would
equally well justify hundreds of other library compounds, which means it is not evidence
about this one.

{common_rules}

Target:
{target_block}

Mechanism under study:
{mechanism_block}

Evidence cutoff: {cutoff_date}.

Candidate under debate:
{candidate_block}

Evidence graph so far:
{graph_block}

Debate so far:
{transcript_block}

Tools available:
{tool_block}

Respond with this exact schema:
{{"tool_calls":[{{"tool":"<name>","args":{{}}}}],"new_nodes":[{{"id":"<new id>","claim":"<=40 words","source":"<tool or record>","confidence":<0..1>}}],"new_edges":[{{"s":"<id>","d":"<id>","k":"support|refute|entail","why":"<=25 words"}}],"argument":"<=120 words","non_specificity_check":"<would this argument justify most of the library? yes/no + why, <=30 words>","recall_flag":"<empty string, or what you caught yourself recalling>"}}
"""

PI_PROMPT = """\
You are the Principal Investigator. You do not edit the evidence graph and you do not
argue. You referee, score, and decide whether the investigation continues.

Score each candidate on the evidence actually present, weighing: the balance of support
against refutation, whether the mechanistic chain from compound to target is connected
or has a gap, and whether the supporting paths are genuinely independent of each other
or are the same claim restated. Three restatements of one claim is one piece of
evidence.

A candidate whose entire case rests on `speculative` nodes scores below one with a
single measured value, however unexciting.

{common_rules}

Target:
{target_block}

Mechanism under study:
{mechanism_block}

Evidence cutoff: {cutoff_date}.

Heuristics retrieved from previous investigations:
{heuristics_block}

Candidates and their evidence graphs:
{graphs_block}

Debate this round:
{transcript_block}

Respond with this exact schema:
{{"scores":[{{"id":"<candidate id>","score":<0..1>,"mech_connected":<bool>,"independent_paths":<int>,"rationale":"<=50 words"}}],"graph_actions":[{{"id":"<candidate id>","instruction":"<what the next round must resolve, <=30 words>"}}],"continue":<bool>,"stop_reason":"<empty, or why the investigation should stop>"}}
"""

CREDIT_ASSIGNMENT_PROMPT = """\
You are the Principal Investigator, writing the Credit Assignment Report after an
investigation has concluded.

You are told which candidate was correct. Your job is to explain what in the *process*
led the investigation toward or away from it, and to propose concrete revisions to the
agents' instructions. Do not propose revisions that encode the answer to this specific
case — a revision that names this compound, this target or this disease is worthless,
because the next investigation is a different disease. Propose revisions about how to
reason.

{common_rules}

What the investigation concluded:
{outcome_block}

What was correct:
{truth_block}

Full debate record:
{record_block}

Respond with this exact schema:
{{"diagnosis":"<=150 words","prompt_revisions":[{{"agent":"explorer|proponent|skeptic|pi","add_or_change":"<=60 words","why":"<=40 words"}}],"heuristics":[{{"when":"<condition, <=25 words>","then":"<action, <=25 words>","generality":"case_specific|target_class|general"}}]}}
"""


# --------------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------------


def tool_block(names: list[str] | None = None) -> str:
    """Render the tool catalogue as it appears in a prompt."""
    names = names or list(TOOLS)
    return "\n".join(
        f"- {n}({', '.join(f'{a}: {t}' for a, t in TOOLS[n]['args'].items())}) -> "
        f"{TOOLS[n]['returns']}\n  {TOOLS[n]['description']}"
        for n in names
    )


def blinded_target_block(gene: str, uniprot: str, sequence_len: int, site: str) -> str:
    """Target description with no disease name and no drug name.

    Deliberately omits the full amino-acid sequence: 02's screen prompt omits it too,
    and including it would cost 415-3,300 tokens for information the agents cannot
    reason over directly. The sequence stays reachable through `txgemma_score`.
    """
    return (
        f"Gene: {gene}\n"
        f"UniProt: {uniprot}\n"
        f"Length: {sequence_len} aa\n"
        f"Binding site class: {site}"
    )


def mechanism_block(mech: str, phenotypes: list[str], free_text: str) -> str:
    return (
        f"Required pharmacological action: {mech}\n"
        f"Observed phenotypes: {', '.join(phenotypes)}\n"
        f"Mechanistic description: {free_text}"
    )


def library_block(compounds: list[dict[str, str]]) -> str:
    """`compounds` are {id, smiles}. Names are withheld on purpose: a drug name is a
    direct retrieval key into the model's memorised indication knowledge, which is
    exactly what this arm must not be scored on."""
    return "\n".join(f"{c['id']}\t{c['smiles']}" for c in compounds)


def transcript_block(turns: list[dict[str, str]], max_turns: int | None = None) -> str:
    if max_turns is not None:
        turns = turns[-max_turns:]
    return "\n".join(f"[{t['role']}] {t['argument']}" for t in turns)


def heuristics_block(heuristics: list[dict[str, str]]) -> str:
    return "\n".join(f"- WHEN {h['when']} THEN {h['then']}" for h in heuristics)


def build(
    role: Literal["explorer", "proponent", "skeptic", "pi", "credit"],
    **kw: Any,
) -> str:
    """Assemble a full prompt for one agent turn."""
    templates = {
        "explorer": EXPLORER_PROMPT,
        "proponent": PROPONENT_PROMPT,
        "skeptic": SKEPTIC_PROMPT,
        "pi": PI_PROMPT,
        "credit": CREDIT_ASSIGNMENT_PROMPT,
    }
    return templates[role].format(common_rules=_COMMON_RULES, **kw)
