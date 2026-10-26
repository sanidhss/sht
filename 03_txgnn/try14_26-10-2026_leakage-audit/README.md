# try14 — leakage audit, prompted by TxGemma-predict's voided verdicts

## Why
TxGemma-predict's try08/try09 found that every one of its pre-try09 verdicts came from
a leaky test (all 12 hard negatives were in training) plus an over-trained model (best
epoch was 0 of 3, with no validation set). All of those verdicts were declared void,
**including its Gaucher pass and its CF pass**. Two questions for this track:
1. Does that change anything this track has claimed?
2. Does TxGNN have an analogous leak that `check_blind()` doesn't catch?

## 1. Consequences of TxGemma-predict's voided verdicts for this track
- This track's "decisions" record justified staying graph-only partly by "TxGNN wins
  Fabry while TxGemma-predict wins Gaucher, so the two evidence sources are
  complementary." **The TxGemma half of that is void.** Under its leak-free harness,
  TxGemma-predict passes TTR on 2 of 3 seeds. Its earlier CF pass did not reproduce
  (n=1 so far, with a re-run pending). Gaucher and Fabry have not been re-run yet.
  The "which disease wins depends on evidence source" contrast is therefore
  **unsupported at present** for every disease, not only CF/TTR. The decision to keep
  TxGNN graph-only can stand on its other reason (avoiding risky changes to a buggy
  third-party library), but not on the complementarity finding.
- try13's README said the contrast was "seen for Fabry/Gaucher"; corrected there.

## 2. Is TxGNN leaking? Checked four ways

### (a) Direct edges — clean (already enforced)
`check_blind()` passed on every seed of every run: no drug–disease edge of any
relation type to a held-out node is in train.

### (b) Do validation/test edges reach message passing? — No
The message-passing graph is built from the training split only:
`TxData.py:108`, `create_dgl_graph(df_train, df)`. Validation edges are scored only
for checkpoint selection (`TxGNN.py:271`, `evaluate_fb(... g_valid_pos ...)`).

### (c) Neighbour-ring leakage — `audit_neighbour_leakage.py` / `.out`
`check_blind()` does not look at the next ring out: an indication from the reference
drug to a disease that is graph-close to the eval node. That would put the answer
2 hops away. TTR needed this handled explicitly in try13; nobody had checked it for
Fabry or Gaucher. Results (reference drug → disease, where the target disease is a
direct disease_disease neighbour or shares at least 1 with the eval node):

| target | edge | split | verdict |
|---|---|---|---|
| Fabry | migalastat → *lysosomal lipid storage disorder* (shares *sphingolipidosis*) | **VALID** | Blind **by luck**. The random 5% validation draw happened to take it. Not in message passing. It slightly influences checkpoint selection (1 edge among thousands). |
| Gaucher | miglustat → *Niemann-Pick disease* (shares 4, incl. *sphingolipidosis*) | **TRAIN** | **Real 2-hop path in training.** |
| CF | — | — | clean |
| TTR | — | — | clean (sibling nodes were held out on purpose in try13) |

How to read these:
- **Fabry**: *lysosomal lipid storage disorder* is a parent category that contains
  Fabry, and migalastat's indication on it exists only because of Fabry. If that edge
  had landed in train, it would have been genuine label leakage. It didn't, but only
  because of the random draw. The split cache is seed-independent (try13 note), so all
  3 Fabry seeds share this draw. **The Fabry result is valid as run, but the blindness
  isn't structural.** Any rerun with a fresh split could leak. Fix for future runs:
  add parent-category nodes like this to the holdout list, as was done for TTR.
- **Gaucher**: Niemann-Pick type C is a different disease, and miglustat is genuinely
  approved for it. So this is legitimate prior knowledge, exactly the
  "treats-a-sibling-disease" signal drug repurposing relies on, not a leaked label.
  But it is the most likely explanation for miglustat's ensembled rank of 17, and
  should be stated whenever that number is reported. Untested: re-running Gaucher with
  Niemann-Pick also held out would measure how much the rank depends on it.

### (d) Configuration tuned on the test result — yes, for Fabry
`attention=True` (try07) was added after Fabry's first result (migalastat 2,863rd),
motivated by inspecting migalastat's own answer path (its GLA edge "diluted" among 770
drug_drug edges). That is a decision informed by the test outcome. It's the same
family of problem as TxGemma-predict's, though much milder: one switch, chosen for a
stated mechanistic reason, not repeated tuning. Consequences:
- Fabry's 17 (try10/11) is an **optimistic estimate**. The configuration was picked
  partly because of how it handled this exact pair.
- **CF and TTR (try13) are the only truly prospective tests in this track.** They ran
  with the configuration frozen before their results were seen. Both were weak:
  CFTR modulators 1,208–1,652, tafamidis 974.

The prospective evidence is weaker than the headline Fabry/Gaucher numbers, and the
thesis should say so.

## Bottom line
- No hard leak: no test label reaches training anywhere, and that's verified, not
  assumed.
- Fabry is blind by luck at the parent-category level, and its configuration was
  chosen after seeing the result, so treat 17 as optimistic.
- Gaucher's miglustat result has a legitimate but strong sibling-indication path
  (Niemann-Pick) in training.
- The only fully prospective, leak-audited results (CF, TTR) are weak.
- The cross-model "complementarity" claim is unsupported until TxGemma-predict re-runs
  Gaucher, Fabry and CF through its leak-free harness.

## Suggested follow-ups (not run)
1. Gaucher with Niemann-Pick added to the holdout: measures how much miglustat's rank
   depends on the sibling indication. One 3-seed job, about 8h.
2. Fabry with *lysosomal lipid storage disorder* added to the holdout: should change
   nothing (the edge was never in train). It would make the blindness structural.
   Could be combined with (1).
3. Run `audit_neighbour_leakage.py` as part of the precheck for every new target.
