# Mathematics — proofs, calculations, and open reviews

[Research home](https://github.com/d6g8k5htny-coder/main) · [Topic guide](https://github.com/d6g8k5htny-coder/main/blob/main/docs/RESEARCH_INDEX.md) · [Run the checks](https://github.com/d6g8k5htny-coder/main/blob/main/docs/REPRODUCE.md) · [Claim manifest](claims/LANDING_CLAIMS.json) · [Proof availability](PROOF_INDEX.md) · [Work queue](https://github.com/d6g8k5htny-coder/main/issues/86)

## Read a result

| Topic | Proof or entry point | Review and scope |
|---|---|---|
| Gaussian lifetime coefficient | [SIDE24 dimensions 2 and 3](coefficients/side24_v1/PROOF.md) | [main #65](https://github.com/d6g8k5htny-coder/main/issues/65); conditional on the parent lifetime formula |
| Quantitative lifetime density | [Bounded unrestricted remainder](frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md) | [main #67](https://github.com/d6g8k5htny-coder/main/issues/67); parent interfaces remain under review |
| RN critical-point counting | [Probability-to-count interface](frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md) | Exact implications and counterexamples; the Gaussian triple integral is not evaluated |
| RN fixed-remote height window | [Cubic count and contact kernel](frontiers/remote_window_20260924/PROOF.md) | Actual count bound on a fixed remote region; shrinking spatial cutoffs and witness collisions remain outside scope |
| RN reviewed fixed-annulus height window | [Fixed scaled-annulus theorem](frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md) | [Source-bound nonauthor review](reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md) accepts the fixed d=2 annulus / compact positive-gap / between-pin-window scope only; no global D5/RN/JETMOD closure |
| P15 obstruction covers | [Original-coordinate realization](frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md) | Specified family and cover; 816 versus whole-ground 818 |
| P15 failed extension | [Price-range counterexample](frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md) | Unrestricted same-palette transformed-price extension is false |
| P15 valid price extension | [Low-probability transformed-price budget](frontiers/price_budget_20260924/PROOF.md) | New sufficient hypotheses: demand at least 2 and probabilities at most 1/4 |
| P15 full probability range | [Full transformed-price budget and sharp constant](frontiers/full_price_20260924/PROOF.md) | Demands at least 2; every independent probability vector; same palette; sharp uniform factor 1/[3-log(3e-2)] |
| Downstream hard gate | [Fail-closed promotion control](frontiers/downstream_gate_20260925/README.md) | [main #90](https://github.com/d6g8k5htny-coder/main/issues/90) / [#86](https://github.com/d6g8k5htny-coder/main/issues/86); integrity only; `lemma_closed` stays false |

Each row is governed by its exact statement, source identity, domain, dependencies, and review object in [claims/LANDING_CLAIMS.json](claims/LANDING_CLAIMS.json). Some rows remain author-side or fail-closed; the fixed-annulus height-window row has a scoped nonauthor technical review. A merge or green test never broadens a claim beyond that recorded scope. The counterexample and the restricted successor concern different hypotheses. The full-range successor removes the probability cutoff, not the demand or realized-family hypotheses; the older 16/27 factor remains better on its smaller domain.

## Run a calculation

All commands below use the Python standard library and run from this repository root.

```sh
python -B -S coefficients/side24_v1/coefficient.py
python -B -S -m unittest discover -s coefficients/side24_v1 -p 'test_*.py' -v
python -B -S -m unittest discover -s frontiers/three_fronts_20260924 -p 'test_*.py' -v
python -B -S frontiers/remote_window_20260924/remote_window.py
python -B -S -m unittest discover -s frontiers/remote_window_20260924 -p 'test_*.py' -v
python -B -S frontiers/price_budget_20260924/price_budget.py
python -B -S -m unittest discover -s frontiers/price_budget_20260924 -p 'test_*.py' -v
python -B -S frontiers/full_price_20260924/full_price.py
python -B -S -m unittest discover -s frontiers/full_price_20260924 -p 'test_*.py' -v
python -B -S frontiers/downstream_gate_20260925/hard_gate.py
python -B -S -m unittest discover -s frontiers/downstream_gate_20260925 -p 'test_*.py' -v
```

The full-price package has 36 distinct tests and seven semantic mutation controls. Run `python -B -S frontiers/full_price_20260924/run_validation.py --output /tmp/full-price-new-run` for both modes; choose a new output directory outside the source tree. `--mode normal` or `--mode optimized` permits bounded split runs. Finite probability grids are checks, not the proof of the full-cube statement.

[Full reproduction guide](https://github.com/d6g8k5htny-coder/main/blob/main/docs/REPRODUCE.md) explains normal/optimized runs, mutation controls, and pinned versus current-source replay. [Frontier directory](frontiers/README.md) groups the packages. Older dated source files are retained unchanged; use the review links for subsequent corrections.

## Where the other work lives

[main](https://github.com/d6g8k5htny-coder/main) holds the campaign, reviews and legacy research-branch links. [meta-framework](https://github.com/d6g8k5htny-coder/meta-framework) holds exact source identities; [query-](https://github.com/d6g8k5htny-coder/query-) looks them up; [trial](https://github.com/d6g8k5htny-coder/trial) checks integration; [google-drive](https://github.com/d6g8k5htny-coder/google-drive) carries selected public replicas. Private experiments are not exported through those public routes.
