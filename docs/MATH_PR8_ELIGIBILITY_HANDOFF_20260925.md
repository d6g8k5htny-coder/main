# Math- PR #8 own-node eligibility handoff

**Object:** MATH-PR8-OWN-NODE-ELIGIBILITY-20260925-v1  
**Against:** Math- head `5887a2f8aaf121760362e6e90f9974f1e0078bc5`  
**Author:** Cursor (main write). **Scientific effect: NONE.**

## Diagnosis

[Math- PR #8](https://github.com/d6g8k5htny-coder/Math-/pull/8) implements the
[#90](https://github.com/d6g8k5htny-coder/main/issues/90) / [#86](https://github.com/d6g8k5htny-coder/main/issues/86)
downstream hard gate, but at head `5887a2f8` dependency terminality alone can
still allow an `AUTHOR_SIDE_CANDIDATE` node to become `controlling`. That fails
the own-node eligibility reading of #90: only a positive reviewed disposition
(`PROVED_REVIEWED`) may become controlling, and terminal dependencies remain
separately necessary.

## Exact fix (ready to apply on Math-)

Portable patch: [`patches/math_pr8_own_node_eligibility.patch`](patches/math_pr8_own_node_eligibility.patch)

SHA-256 `1648237df8525276b17704d12a934cc97c12b1f1184af671ee894b65d60816f1` (8156 bytes).

Applies on top of Math- `5887a2f8`:

1. Add `CONTROLLING_ELIGIBLE = {PROVED_REVIEWED}` and enforce it in
   `promotion_allowed()`.
2. Replace fail-open `test_promote_after_all_terminal_deps` with:
   - `test_author_side_with_terminal_deps_cannot_become_controlling` (negative)
   - `test_proved_reviewed_with_terminal_deps_may_become_controlling` (positive)
3. Add mutant `bypass_own_node_eligibility`; raise `EXPECTED_TESTS` to 28.
4. Refresh `RESULTS.json` / `SOURCE_FILES.json` hashes after the gate change.

Local verification (recorded in
[`patches/math_pr8_eligibility_fix_REPORT.json`](patches/math_pr8_eligibility_fix_REPORT.json)):

```text
28 tests, 8 distinct semantic mutations, modes normal+optimized → passed
```

## Why this is filed on main

Cursor Cloud Agent can write `main` but received **403 pushing to
`d6g8k5htny-coder/Math-`** (`Permission denied to cursor[bot]`). The repair was
prepared and locally verified in a Math- worktree; it could not be published on
that forge tip from this session. This note + patch are the source-bound
eligibility recipe for an agent/owner with Math- write access (or for merging
into Math- #8).

## Apply recipe

```sh
git clone --filter=blob:none https://github.com/d6g8k5htny-coder/Math-.git
cd Math-
git checkout 5887a2f8
git apply /path/to/main/docs/patches/math_pr8_own_node_eligibility.patch
python -B -S frontiers/downstream_gate_20260925/hard_gate.py
python -B -S -m unittest discover -s frontiers/downstream_gate_20260925 -p 'test_*.py' -v
python -B -S frontiers/downstream_gate_20260925/run_validation.py --output /tmp/downstream-gate-eligibility
```

Then push to the Math- #8 branch (or open a successor PR) from a credential that
can write Math-.

## Non-claims

- Does not merge Math- #8 or main [#92](https://github.com/d6g8k5htny-coder/main/pull/92).
- Does not flip any scientific Boolean; `lemma_closed` stays false.
- Does not edit Math- PR #7 or award R17 independence credit.
- Does not touch the #91 Drive vault.
- Installing the patch does not promote mathematics; it only closes an integrity hole.
