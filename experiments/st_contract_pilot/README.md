# ST contract pilot — proposed software package

This source-bound pilot checks a finite algebra/representation contract. It does
not verify a random-field theorem, authenticate a remote artifact, admit evidence,
change scientific status, or implement the complete Universality Compiler.
It does not validate every descriptive field: `scope.observable` and explanatory
formula strings are not consumed as proof of the intended theorem.
This layout is a local adoption proposal, not an installed or merged component.

## Run from the repository root

```sh
python -I -B -S tests/test_st_contract_pilot.py
python -I -B -O -S tests/test_st_contract_pilot.py
```

The four adapter tests verify frozen bytes, execute the 28-method overlay suite,
execute the 32-method scaling suite, and inspect all eight case observations.
Children use the same interpreter and normal/optimized mode, with isolated import
paths, disabled bytecode and no site packages. The outer adapter imports no pilot
modules into the discovery process. The check scripts are deliberately named
`check_*.py`; their tests are executed by the adapter, not removed or skipped.

For the report only, run `python -I -B -S
experiments/st_contract_pilot/run_cases.py` (on one line); the optimized variant is
also supported. It reports exact algebra and refusal observations, never scientific
acceptance. This command-only runner assumes the delivered trusted package; the
adapter verifies all eight file identities before launching any child.

## Source preservation

`SOURCES.json` binds the original public deliveries and the exact package bytes.
The diagnostic, two dependencies and two R2 fixtures are unchanged. The scaling
check is only renamed. The overlay check has one dependency-path edit, and the
case runner has one check-script-name edit. No mathematical or expected-result
change is made. The arithmetic snapshot is byte-identical to the existing
`fold_controls.py` at the recorded source commit, not a new arithmetic derivation.
Frozen copies keep this pilot reproducible; changing a pin needs actual source
and dependency review, not a timestamp refresh.

## Why subprocess isolation matters

The original standalone scaling check intentionally installs its private modules
in `sys.modules`. A naive in-process repository import overwrote existing
`fold_controls`, `st_object_overlay`, and `st_scaling_diagnostic` entries in the
preflight experiment. Isolation avoids that cross-test contamination without
rewriting the tested modules. This observation concerns a proposed integration,
not a defect in the original standalone delivery's declared scope.

Hash mismatches fail immediately before launch. Do not replace the integrity
assertions with `unittest.subTest`: a failed subtest is recorded but execution
continues. Preflight negative controls verify that corrupt dependencies launch
no child, zero-test success is refused, and a matching summary with failed exit
is refused. These checks do not provide hostile-process containment or protection
against concurrent filesystem mutation. Inputs must be a trusted stable checkout.

## Discovery and hosted coverage are different

The adapter is discoverable by the ordinary `tests/test_*.py` discovery command.
At the inspected repository reference, the navigation workflow uses that command
but does not trigger on these new paths alone. The universality workflow's path
filter does not match this sibling directory; its separate fifty-law campaign is
unchanged. Broad workspace/public-shop checks run on PRs but their green outcomes
must not be relabeled as execution of this pilot. No hosted pilot result or full
repository test pass is established by this proposal. Explicit hosted coverage
and current composition remain adoption/integration prerequisites to reconcile.

## Review and provenance

Current component review records live on main issue 275. The original R2 data
review does not cover the overlay, scaling implementation or this new adapter.
The proposal author is OpenAI / GPT-6 Astra Pro, continuing ULHI foreground work;
source-exposed, with no organizational-independence or scientific-acceptance credit.
The scope and handoff are main#275/6357059903. No workflow, protection, register,
source-map cell, schedule or other actor's branch is changed by this package.
