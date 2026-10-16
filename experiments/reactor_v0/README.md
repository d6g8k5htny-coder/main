# Reactor v0 — finite amplitude-transform control freeze

This directory freezes one deterministic, engineering-only regression control.
It does **not** sample a random field, run a campaign, consume a Gaussian or
asymptotic theorem conclusion, alter a scientific status, or execute Reactor B.

The 4×4 periodic integer grid in `fixture.json` has positive finite ordinary
superlevel H0 bars `[[1, 0], [8, 0]]` and essential birth `27`. The essential
class is excluded. For the declared whole-field amplitude law parameters
`alpha = 2/3`, `a = 2`, and `t = 1/4`, the exact exposed regression control is

```text
sum_i min((t / (a * lifetime_i))**alpha, 1) = 1/4 + 1/16 = 5/16.
```

Only rational powers whose required rational roots are exact are supported;
other inputs raise `ValueError`. The counted event is
`A * (birth - death) <= t` (inclusive), so every positive finite bar contributes
`min((t / (a * lifetime))**alpha, 1)`; the inclusive selector
`short_bar_lifetimes` applies to a fixed realised barcode. Expected answers are
exposed regression controls, not blind discoveries.

`source_manifest.json` allowlists the two open parent proof commits and records
raw-byte SHA-256 separately from Git blob membership. Both entries explicitly
say that no theorem conclusion is consumed. `closure.json` pins every consumed
helper, fixture, test, validator, runner, mutation harness, README, manifest,
and path-scoped workflow by raw SHA-256; it intentionally cannot hash itself.

Run the complete frozen controls from the repository root:

```sh
python3 -B -m unittest discover -s experiments/reactor_v0 -p 'test_*.py' -v
python3 -B -O -m unittest discover -s experiments/reactor_v0 -p 'test_*.py' -v
python3 -B experiments/reactor_v0/mutation_harness.py
python3 -B -O experiments/reactor_v0/mutation_harness.py
python3 -B experiments/reactor_v0/run_control.py
python3 -B -O experiments/reactor_v0/run_control.py
```

The workflow has read-only repository permissions, is triggered only by this
directory or its own workflow file, and ends with a clean-worktree check so the
run stage cannot silently edit the evaluator, tests, or any other tracked input.

Canonical coordination: [Reactor A pickup](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6204609205),
[owner mandate](https://github.com/d6g8k5htny-coder/main/blob/b665307520948308a53d190d7341bf301df3b3a2/governance/OWNER_MANDATE_20261015.md),
and [standing delegation](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-AUTONOMY-20260923-v2.1.md).
