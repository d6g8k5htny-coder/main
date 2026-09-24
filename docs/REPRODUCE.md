# Reproduce the work

[Home](../README.md) · [Research guide](RESEARCH_INDEX.md) · [Workspace](WORKSPACE.md)

## Mathematical sources

Clone [Math-](https://github.com/d6g8k5htny-coder/Math-) and run these commands from its root. The named packages use the Python standard library. Record the actual commit with `git rev-parse HEAD` and the interpreter with `python --version` before comparing results.

```sh
git clone https://github.com/d6g8k5htny-coder/Math-.git
cd Math-
git rev-parse HEAD
python --version
```

| Work | Read | Execute from Math- root |
|---|---|---|
| Coefficient | `coefficients/side24_v1/PROOF.md` | `python -B -S coefficients/side24_v1/coefficient.py` |
| Lifetime/RN/P15 package | `frontiers/three_fronts_20260924/README.md` and its separate price-boundary note | `python -B -S -m unittest discover -s frontiers/three_fronts_20260924 -p 'test_*.py' -v` |
| New transformed-price budget | `frontiers/price_budget_20260924/PROOF.md` | `python -B -S frontiers/price_budget_20260924/price_budget.py` |

Run all mathematical test groups in both modes:

```sh
for mode in normal optimized; do
  opt=""; [ "$mode" = optimized ] && opt="-O"
  python -B $opt -S -m unittest discover -s coefficients/side24_v1 -p 'test_*.py' -v
  python -B $opt -S -m unittest discover -s frontiers/three_fronts_20260924 -p 'test_*.py' -v
  python -B $opt -S -m unittest discover -s frontiers/price_budget_20260924 -p 'test_*.py' -v
done
```

The source snapshots contain 30 coefficient tests, 57 three-front tests (54 core plus 3 supplementary), and 26 price-budget tests. Repeating modes or runtimes does not add distinct tests or independent analytic review. A failed baseline must not be hidden among expected mutation failures.

## Deliberate error controls

Each package runner requires a NEW output directory outside its source directory. For example, choose a fresh path for each invocation:

```sh
python -B -S frontiers/price_budget_20260924/run_validation.py --output /tmp/price-budget-run-001
python -B -S frontiers/three_fronts_20260924/run_validation.py --output /tmp/three-fronts-run-001
```

The new runner checks 26 tests and five deliberately incorrect code variants in normal and optimized modes. The older three-front runner retains its 54-test core and ten mutations; run `test_price_boundary` as well, as the discovery command above does. The runners do not prove the Gaussian continuum estimates or supply nonauthor review.

## Exact cross-repository lookup

Keep sibling checkouts named `Math-`, `meta-framework`, `query-`, and `google-drive`. From their common parent:

```sh
python -B -S query-/research_query.py --registry meta-framework/registry.json --key lifetime-remainder
python -B -S query-/research_query.py --registry meta-framework/registry.json --key p15-price-budget
python -B -S query-/research_query.py --registry meta-framework/registry.json --verify --workspace .
```

The read-only lookup has no network or retrieved-code execution. It refuses unknown keys, invalid identities and altered local bytes. A legitimate later edit also fails an older hash: inspect the newer source and catalog instead of disabling the check. The current catalog is a curated subset, not a complete inventory of every Drive file or historical branch.

## Current checks versus pinned historical replays

[trial's federation replay](https://github.com/d6g8k5htny-coder/trial/blob/main/federation/replay.py) deliberately downloads the exact original coefficient-era commits. That preserves a reproducible historical experiment; it does NOT automatically run later price-budget or lifetime additions. Run the current Math- tests above for those additions. A skipped federation fixture in a trial-only checkout is not a successful multi-repository replay.

For legacy RN/24-jet and other hardening work, use the [execution guide on that branch](https://github.com/d6g8k5htny-coder/main/blob/chatgpt/drive-github-hardening-20260919/docs/RESEARCH_EXECUTION.md). Do not run a navigation-only default checkout and label it a full legacy research test.

## Navigation checks

From the default checkout of this `main` repository:

```sh
python -B -S tools/workspace_landing_check.py
python -B -S tools/navigation_check.py
python -B -S -m unittest discover -s tests -p 'test_*.py' -v
```

The first check retains the historical byte-custody and declared local-link scope. The second additionally checks fragments in declared navigation pages; `--verify-public` explicitly enables a bounded check of the named public source targets. It does not crawl every branch or Drive link, validate credentials, execute retrieved sources, or certify mathematics. Live source drift is a reason to inspect a changed target, not to call its mathematics false.
