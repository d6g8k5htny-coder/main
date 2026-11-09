# QP-C54 tranche 11a: what landed on the default-branch tips after the accounting cuts (9 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. No repository code was run.

## Why

Tranches 1–8 censused Math- at `3b11bff6`/`c88c34b2` and main at `d2b0216e`. After those cuts the tips moved:
- Math- by 69 commits to `f532c3e2`, merging #536, #541, #548 and #550–#554;
- main by 130 commits to `5ebe5771`.

This tranche accounts for what landed since.

## Scope (`tip_delta.py`)

For each repository, the census takes the mathematical-type files the default branch added or changed between the old cut and the
new tip. It excludes root files, the engineering directories of T8 (plus `recovery/` and `incoming/`), and any file whose blob or
SHA-256 is named by a QP-C54 tranche file, catalog candidate v2 or the CLOSED index (`32ab7d90` and `fdb225c2`).

**Math-** (`math_delta.json`): 76 files, 54 of them already bound. 22 are unbound.
- 6 are new verifier and test bytes in `frontiers/extreme_cluster_sampling_20260930` and `frontiers/radial_rate_20260930`. These
  directories are covered by the locations of adopted Q-1099 and Q-1098 (the T8 rule), so they are not reclassified.
- The other 16 form two objects.

**main** (`main_delta.json`): 2 files, both in `docs/cbz/`, forming one object.

`facts/dbatch1.json` holds the three objects.

## Pipeline

- Workflow `workflow_t11a.js` ran a draft and an adversarial screen (2 agents), which fixed 3 problems.
- `../qp_c54_t10_unmerged_math_refs_20261102/normalize_verdicts_t10.py` made 0 changes.
- `assemble_t11a.py` is `assemble_t10.py` with per-object repositories and tip bindings, and the same fail-closed checks. Rerunning
  it reproduces `records_verified_t11a.json` byte for byte.
- The manual checks are clean (`manual_checks_output.txt`).

## Result (`records_verified_t11a.json`)

**0 obligations; 11 items.**

| Object | Items | Classification |
|---|---|---|
| Math- `frontiers/determinant_moment_20261025` (12 files: ledgers, governance, status, tooling, manifest, mutation runners, probes) | 7 | NOT_OBLIGATION |
| Math- `reviews/elder_selection_chain_audit_20261031` (#554, merged) | 2 | REVIEW_OF (AUDIT.md, CLAIMS.json) and NOT_OBLIGATION (its checker and test) |
| main `docs/cbz/CANONICAL_AUDITORS.md`, `G11_AUDITOR.md` | 2 | NOT_OBLIGATION |

- The determinant `STATUS.md` at the tip asserts that "the 66 exact conditional companion statements are CLOSED by actual
  source-qualified native4871 execution". This item records that as the source's own assertion (A7).
  - The CLOSED index has K-DM-001 to K-DM-066 bound at Math- `74565f01` with receipt `cbz/evidence/kernel/determinant/receipt.json`,
    not the 4871 run.
  - No Lean file of the package changed since the cut.
- The elder-selection audit assigns one of the owner's five status values to 23 claims of the Theorem A/B/C and lower-bound
  chains, and lists 37 open obligations. It proves nothing: "No theorem is proved, improved, weakened or withdrawn here."
  - Its refuted entries rest on the Math-#543 one-law countermodel, which tranche 10 accounts as QP-C54-T10-025.

## Limits

- Both agents are Anthropic Claude.
- Root files, including the canonical `STATUS.md` and `PROOF_INDEX.md`, and the engineering directories are out of scope, as in
  tranche 8.
- **Reproduce:**
  1. `python3 -I -B tip_delta.py MATH c88c34b2 f532c3e2 BOUND_DIRS... OUT`, with `BOUND_DIRS` the QP-C54 retention directories, the
     candidate's directory and a directory holding both CLOSED indexes;
  2. the same for main with `d2b0216e` and `5ebe5771`;
  3. `python3 -I -B assemble_t11a.py CLONES_JSON facts CANDIDATE_V2 OUT verdicts_normalized_t11a.json`.
