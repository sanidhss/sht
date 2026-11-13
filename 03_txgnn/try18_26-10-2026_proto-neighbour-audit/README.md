# try18 — prototype-neighbour audit: Fabry's answer leaks in through a near-duplicate node

## Why
try17 found Fabry's ranking puts every rare-disease drug near the top. The question
was whether that's a leak. try14's audit had only checked drug edges to
disease_disease neighbours. It never checked the path the model actually uses for
held-out diseases: TxGNN's **prototype** step.

## The mechanism (read from `txgnn/model.py` ~L76–260, `utils.py` L943, L985, L995)
With `proto=True, proto_num=3, sim_measure='all_nodes_profile', agg_measure='rarity'`
(our config since try07):
- profile(d) = one-hot of d's disease_disease neighbours ++ one-hot of its proteins,
  taken from the **training graph G** (`obtain_disease_profile`).
- Keys = diseases with ≥1 edge of the scored relation (indication) in G.
- A disease's scoring representation = (1−c)·own + c·Σ wₖ·keyₖ over its 3
  cosine-most-similar keys. w is the L1-normalised similarity, and
  c = 0.7·exp(−0.7·deg) + 0.2.
- A held-out disease has deg = 0, so **c = 0.9: 90% of its representation is the blend
  of its 3 nearest keys.** TxGNN's zero-shot design relies on this. It also means
  whatever drugs those 3 keys are indicated for get pulled to the top.

`proto_neighbours.py` / `.out` reproduces this on CPU from each run's actual training
split.

## Result

| held-out disease | its 3 prototype keys (cos, share of representation) | key indications | true drug among them? |
|---|---|---|---|
| **Fabry** | **chylomicron retention disease** (0.69, 31%) · lysosomal lipid storage disorder (0.66, 30%) · Wolman disease (0.64, 29%) | **migalastat**, alipogene tiparvovec · sebelipase alfa · sebelipase alfa | **YES (migalastat)** |
| Gaucher | **Niemann-Pick** (0.25, 36%) · Hurler (0.19, 27%) · Fabry (0.19, 27%) | **miglustat** · laronidase · migalastat | **YES (miglustat)** |
| Gaucher, NP held out (try15) | Hurler · Fabry · cerebrotendinous xanthomatosis | laronidase · migalastat · cholic acid | no |
| CF | idiopathic bronchiectasis (0.23) · bronchiectasis (0.22) · altitude sickness (0.20) | acetylcysteine, cysteine, β-agonists, catecholamines · same · acetazolamide | no CFTR modulator |
| TTR (ATTRV122I) | lung neoplasm (0.035) · dysthymic disorder (0.034) · neurotic disorder (0.033) | chemo · antidepressants · antidepressants | no |

### Fabry: a real leak, through a PrimeKG data error
In PrimeKG, *chylomicron retention disease* (MONDO 9528) has **all 23 of Fabry's
disease_protein links, including GLA**, plus APOC2 and LPL, and an **indication
for migalastat** (DrugBank-sourced row, `kg.csv` 389339). Real chylomicron retention
disease is a SAR1B lipid-transport disorder with no GLA link and no migalastat use.
The node is effectively a near-duplicate of Fabry that carries Fabry's answer, and it
stays in training when Fabry is held out. Likely cause (**unverified**): chylomicron
retention disease's synonym is "Anderson disease", and Fabry's is "Anderson–Fabry
disease". A name-based mapping in an upstream source probably merged their
annotations.

Consequence: **Fabry's rank of 16 (try10/11) is not a blind result.** The prototype
step copies a leaked migalastat label into 31% of Fabry's representation. Fabry was
the track's headline; it has to be re-run with chylomicron retention disease held out
too (and lysosomal lipid storage disorder, the parent from try14) before any Fabry
claim stands.

### Gaucher: the Niemann-Pick sibling is the #1 prototype
This confirms try14's concern at the mechanism level: 36% of Gaucher's representation
is Niemann-Pick, whose only indication is miglustat. try15 (running) removes it. The
reproduction above predicts that try15's prototypes are Hurler/Fabry/CTX, which
contain no Gaucher drug.

### Why rankings cluster by category (try17), and why migalastat ranks 27 for Gaucher
Fabry is one of Gaucher's prototype keys, so migalastat is pulled up for Gaucher. The
lysosomal neighbourhood (sebelipase, laronidase and other ERT indications) explains the
shared "rare/lysosomal" list.

### CF and TTR
- CF's prototypes are bronchiectasis diseases. Their drugs include CF's own
  acetylcysteine and cysteine, but no CFTR modulator, so the modulators get no
  prototype boost.
- **TTR's prototypes are effectively random.** ATTRV122I's profile has only 3 entries,
  and the best cosine is 0.035. So 90% of its representation is an arbitrary blend of
  lung cancer and depression, which is why chemotherapy and antidepressant-adjacent
  drugs top its list.

## What this means for "why Fabry/Gaucher work and CF/TTR don't"
The dominant factor is **what the 3 prototype diseases are indicated for**:
- Fabry: the answer, via a mislabelled near-duplicate. That's a leak.
- Gaucher: the answer, via a real sibling. Legitimate, but borrowed.
- CF: related diseases, but not the drug class that matters.
- TTR: unrelated diseases, because its node is too sparse to find real neighbours.

## Changes made
- try16 `analyze.py` (its job hasn't run yet) now computes `proto_hit`: whether any
  true drug is indicated for one of the disease's 3 prototype keys. It's reported
  separately and included in the rank regression. Without it, the connectivity test
  would confound "well connected" with "has a neighbour holding the answer".
- try14's audit is superseded on this point: it only checked disease_disease
  neighbours and missed protein-similarity prototypes.

## Framing: the similarity design is sound; these are its caveats
Borrowing from similar diseases is TxGNN's intended zero-shot design (Huang et al.,
*Nature Medicine* 2024). It's not a flaw in itself. What it implies:
1. **Blindness must cover the prototypes.** Hiding the named disease isn't enough: a
   prototype carrying the answer (a data error like Fabry's duplicate, or a real
   sibling like Gaucher's Niemann-Pick) breaks blindness. Check the 3 prototypes.
2. **Inherits data errors.** One mislabelled node can be ~30% of a target's
   representation.
3. **Class-level, not drug-level.** Diseases with shared prototypes get near-identical
   rankings (try17: Fabry–Gaucher 0.957). It can point to a drug class, not
   discriminate within it.
4. **Can't surface disease-unique drugs.** If the right drug is approved only for the
   target, no prototype carries it (CF: CFTR modulators). It favours borrowed over
   novel answers.
5. **Sparse diseases get noise silently.** The 0.9 weight applies whether the best
   cosine is 0.69 (Fabry) or 0.03 (TTR). Report the best prototype cosine as a
   reliability flag.
6. **Seed ensembling doesn't fix it.** Prototypes are deterministic given the graph and
   split, so all seeds share them.
7. **Depends on what else is held out.** Held-out diseases can't serve as prototypes
   (try15 changes Gaucher's; in try16 the 120 can't serve for each other).

## Decision (26-10-2026)
The user decided **not** to run a separate Fabry re-run. try15 (Gaucher without
Niemann-Pick) tests the same mechanism: does the rank survive when the prototype that
carries the answer is removed? Fabry's 16 should be reported with the
leaked-prototype caveat, not as a blind result.

## Recommended next (not run)
1. ~~Fabry re-run~~: declined by the user, see Decision above.
2. Add the prototype check to every target's precheck: compute the 3 keys and look
   for the true drug among their indications. It's cheap, CPU only.
