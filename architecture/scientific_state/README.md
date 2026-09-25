# Scientific state architecture — schema pilot

**Object:** SCIENTIFIC-STATE-SCHEMA-20260925-v1  
**Issue:** [#95](https://github.com/d6g8k5htny-coder/main/issues/95)  
**Scientific effect: NONE.** Installing this architecture package defines a
versioned schema, authority map, ID crosswalk, and verification-level
vocabulary only. It does not discharge premises, promote claims, flip
`lemma_closed`, alter Math- classifications, or accept any theorem. Hash
identity, green CI, and LLM agreement remain non-discharge.
Formal-verification levels are evidence metadata, not acceptance.

## Why this is not a second status store

| Authority | Owns | Path |
|---|---|---|
| Claims firewall | grades, premise openness, composition firewalls | `claims/graph.json` |
| Math- downstream gate | integrity classifications, CONTROLLING promotion | Math- `GRAPH.json` ([PR #8](https://github.com/d6g8k5htny-coder/Math-/pull/8)) |
| Human D0 crosswalk | D0/D4/D5 narrative classification | `docs/DOWNSTREAM_RN_CROSSWALK_20260925.md` ([PR #97](https://github.com/d6g8k5htny-coder/main/pull/97), supersedes #92) |
| Math status packet | `lemma_closed` and related display Booleans | `docs/math_status/` |
| **This package** | schema contract, ID pointers, L0–L5 vocab | `architecture/scientific_state/v1/` |

If a row needs a status, it points at the owning file. It does not duplicate it.

## Files

- [`v1/SCHEMA.json`](v1/SCHEMA.json) — field contract; lists forbidden owned fields
- [`v1/AUTHORITY_MAP.json`](v1/AUTHORITY_MAP.json) — explicit ownership
- [`v1/ID_CROSSWALK.json`](v1/ID_CROSSWALK.json) — pilot ID bridge (no copied statuses)
- [`v1/VERIFICATION_LEVELS.json`](v1/VERIFICATION_LEVELS.json) — L0–L5 evidence metadata

## Checker

```sh
python3 tools/scientific_state_check.py
python3 -m unittest tests.test_scientific_state -v
```

The checker only **refuses** malformed architecture files, missing authorities,
dangling main IDs, or smuggled status/grade/classification/controlling payloads.
It does not compute promotions or reverse-impact closures.

## Deferred (not this PR)

- Fail-closed promotion engine — stays in Math- PR #8; do not reimplement here
- D0 crosswalk / nav / math_status pages — already on tip via #97; do not rewrite here
- Status-bearing populated pilot graph — later, still as pointers or read-only mirrors
- Formal L2/L4/L5 pilot lemma — later, small algebraic lemma from the downstream queue
- Wiring into a global `run_checks.py` — optional follow-up

## Non-claims

- Does not edit `claims/graph.json`, Math-, tip crosswalk/handoff docs from #97, `OPEN_PROBLEMS.md`, or the vault
- Does not flip any scientific Boolean
- Does not award independence credit
