# Phase15 — Local scalar certificates and globally compatible palettes

This package contains four author-side mathematical arguments, exact finite companions, source/priority records, and deployment instructions. It does not solve the unrestricted discrete-convexity conjecture and contains no external mathematical approval. q0/Claude sources and scientific statuses are unchanged.

## Read in order

1. `CURRENT_STATE.md` and `CLAIM_REGISTRY.json` for precise scopes.
2. `proofs/01_CRITICAL_THRESHOLD_CROSSWALK.md`: identify the scalar optimization with the established critical threshold value; strict endpoint translation.
3. `proofs/02_PALETTE_SEPARATED_LOCALIZATION.md`: combine actual local restrictions by an exactly checked cross-witness palette condition.
4. `proofs/03_CONNECTED_UNBOUNDED_WIDTH_FAMILY.md`: an all-rank connected family with a fixed816 palette, nonempty obstruction, and unbounded GLOBAL critical threshold.
5. `proofs/04_COMPLETE_TRANSVERSAL_GRAPH_COMPOSITION.md`: arbitrary graph coupling of scalar modules, with a fixed256L palette and a rigorously allocated hazard budget.
6. `REVIEW_PACKET.md` before using a result as a premise.

`inputs/` contains exact predecessor proof bytes, plus explicit checker examples. The previous P14 all-size scalar proof is an author-side premise, not silently upgraded to independent acceptance. The P11 graph input is the ATTACHED128-color version at its explicit hash, not its distinct64-color sibling.

## Reproduce

```sh
python code/reproduce.py --output /tmp/p15-full
python code/certify_palette_plan.py inputs/PALETTE_THREE_BLOCKS.json
```

For tool runtimes with a strict per-call limit, execute separate batches:

```sh
python code/reproduce.py --output /tmp/p15-batches --batch 0
# Repeat for --batch 1, 2, 3, 4, 5, 6.
python code/reproduce.py --output /tmp/p15-batches --collect
```

The default batch size is6;38 expected jobs require seven batches. Collection verifies every expected job, source/input hashes, output hashes, exit codes, structured status and stderr, then compares ordinary and optimized baseline output. An interrupted batch is not a complete reproduction.

The checker uses exact rational arithmetic and explicit exceptions, not removable assertions. No third-party Python package is needed. It is a finite implementation companion, not a proof assistant. The finite test count includes individual subsets/identities, not independent reviews.

## Certificate checker scope

`certify_palette_plan.py` accepts an explicit minimal-witness antichain, a true coordinate partition, integer palettes, and actual local proper-coloring or scalar-sandwich certificates. It checks all crossing witnesses. Local scalar verification enumerates all subsets and is capped at16 local vertices in this implementation. The mathematical theorem has no such cap. Large implicit families require a proved structural certificate; the721,977,344-coordinate example is NOT enumerated.

No polynomial-time discovery algorithm is asserted. Existing coloring, fractional-cover, critical-threshold and prefix-rounding theories are credited in the source register. Historical novelty of the precise new cover-transfer conclusions remains unestablished.
