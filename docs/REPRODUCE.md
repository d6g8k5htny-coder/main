# Reproduce the work

[Home](../README.md) · [Research index](RESEARCH_INDEX.md) · [Workspace](WORKSPACE.md)

Commands below are routes. A passing run is engineering evidence for the
command that ran. It does not discharge `OBL-H5-JETMOD` or `D3-LEMMA-RN-UNIF`.
`lemma_closed`, `prizes_solved`, `discharges_OBL_H5_JETMOD`, and
`certified_C_H` stay false. `inventable_attempt_accepted` stays false.
SIDE24 carriers marked ABSENT stay ABSENT. Quarantine is not a source of truth.

## This branch

This file lives on `chatgpt/drive-github-hardening-20260919`, not on `main`.
The research checkers and their boundaries are in
[RESEARCH_EXECUTION.md](RESEARCH_EXECUTION.md) and
[VERIFICATION_RUNNER.md](VERIFICATION_RUNNER.md). Record `git rev-parse HEAD`
and the interpreter version with the result. Navigation checks in the last
section do not stand in for that CI.

Inventable probe honesty and the SIDE24 prep walls are in
[the probes README](math_status_probes/README.md). `REFUSED` ≠ discharge.
`EMPTY`, `ABSENT`, `PARTIAL`, and `REFUSED_NOT_24JET` stay honesty labels.
Instrumentation STATUS is `PARTIAL` / `REFUSED_NOT_24JET` only. Objects
marked **ABSENT** in [RN_SIDE24.md](RN_SIDE24.md),
[RN_SIDE24_DENSITY.md](RN_SIDE24_DENSITY.md), and
[RN_SIDE24_CELL.md](RN_SIDE24_CELL.md) stay **ABSENT**. A green probe or
SIDE24 checker is not a source of truth and does not discharge
`OBL-H5-JETMOD`.

## Mathematical sources

Clone [Math-](https://github.com/d6g8k5htny-coder/Math-) and run these commands
from its root. The named packages, as described by those notes, use the Python
standard library. Record the commit you actually checked out. Counts below are
the counts written in the navigation notes being ported. Re-count them on that
commit. This port does not re-certify them.

```sh
git clone https://github.com/d6g8k5htny-coder/Math-.git
cd Math-
git rev-parse HEAD
python --version
```

| Work | Read | Execute from Math- root |
|---|---|---|
| Coefficient | [PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/main/coefficients/side24_v1/PROOF.md) | `python -B -S coefficients/side24_v1/coefficient.py` |
| Lifetime/RN/P15 package | [LIFETIME_REMAINDER.md](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md), [RN_COUNT_INTERFACE.md](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md), [P15_REALIZED_COVERS.md](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md), [P15_PRICE_BOUNDARY.md](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md) | `python -B -S -m unittest discover -s frontiers/three_fronts_20260924 -p 'test_*.py' -v` |
| Transformed-price budget | [PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/price_budget_20260924/PROOF.md) | `python -B -S frontiers/price_budget_20260924/price_budget.py` |

The ported notes recorded 30 coefficient tests, 57 three-front tests, and 26
price-budget tests. Repeating modes does not add independent analytic review.
A failed baseline stays visible.

## Deliberate error controls

Each package runner described in those notes requires a new output directory
outside its source directory:

```sh
python -B -S frontiers/price_budget_20260924/run_validation.py --output /tmp/price-budget-run-001
python -B -S frontiers/three_fronts_20260924/run_validation.py --output /tmp/three-fronts-run-001
```

Those runners exercise tests and deliberately incorrect variants. They do not
prove the Gaussian continuum estimates and they do not supply nonauthor review.

## Exact cross-repository lookup

The ported notes keep sibling checkouts named `Math-`, `meta-framework`,
`query-`, and `google-drive`. From their common parent:

```sh
python -B -S query-/research_query.py --registry meta-framework/registry.json --key lifetime-remainder
python -B -S query-/research_query.py --registry meta-framework/registry.json --key p15-price-budget
python -B -S query-/research_query.py --registry meta-framework/registry.json --verify --workspace .
```

That lookup, as described there, refuses unknown keys and altered local bytes.
A later legitimate edit also fails an older hash. Inspect the newer source.
The catalog is a curated subset.

[trial's federation replay](https://github.com/d6g8k5htny-coder/trial/blob/main/federation/replay.py)
downloads pinned historical commits. It does not automatically run later
additions. A skipped fixture is not a successful multi-repository replay.

## Navigation checks

From this hardening checkout:

```sh
python3 -B -S tools/navigation_check.py
python3 -B -S -m unittest tests.test_navigation -v
```

`tools/navigation_check.py` checks declared inline Markdown links and ATX
fragments on the pages in [NAVIGATION.json](NAVIGATION.json).
`--verify-public` is an explicit opt-in readback of the six named public
Math- files in that manifest. It does not crawl every branch or Drive link,
execute retrieved sources, or certify mathematics. A changed public byte is a
reason to inspect the target. It is not a verdict that the mathematics is false.

The historical main-landing custody checker and `history/2025` byte identities
are absent on this branch. [LANDING_MANIFEST.json](LANDING_MANIFEST.json)
records that absence. It is not a substitute source of truth.
