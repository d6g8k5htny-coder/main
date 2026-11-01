# Proposed §11 quarantine records for 52 Math- proof packets (QP-C54)

By `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0; scientific effect
NONE. These are PROPOSED records (`QP-C54-001` to `QP-C54-052`) for main#695's
`cbz/quarantine/catalog.json`. They close and admit nothing. They cover exactly the 52 packets that
`../math_coverage_gap_20261031/` finds with no disposition at Math- `8151f592918a845bf19eb8eafaf8444540a32734`.

- **`records_verified.json`**: the proposal, in the catalog's `quarantines` schema (`status`
  QUARANTINED, `proposal_status` PROPOSED, `owner` none, `maintenance` zero).
  - Extra fields carry the source evidence: `source_statements` (verbatim `PROOF.md` lines with line
    numbers), `source_hypotheses`, `source_dependencies`, `source_self_limits`, `finite_checks`,
    `floating_point` (USED / NONE / NO_SCRIPTS) and `source_binding` (every packet file's blob,
    sha256 and bytes).
  - The top-level `verification` object records how the records were made and checked.
- **How the records were made:**
  1. One drafting pass (`workflow_draft.js`).
  2. Two independent adversarial passes per batch (`workflow_verify.js`): a fidelity pass, then a
     Layer A pass on its output. No verifier checked a record it wrote. The passes fixed 186 and
     77 problems; 19 records needed no Layer A change.
  3. Then `assemble.py`, which requires every quoted statement to equal its `PROOF.md` lines byte
     for byte at the cut, every packet exactly once, and a classified floating-point field. It
     takes every binding from `facts.json`.
  4. Manual checks: NO_SCRIPTS against the presence of Python files for all 52, NONE records with
     float indicators read by hand, and a lint for overclaiming in our own voice.
- **Floating point:** NONE 30, USED 10, NO_SCRIPTS 12. Per A3, USED output is never certification.
- **Independence:** all agents are Anthropic Claude, the same family as the author. This is not
  a scientific review.
- **Reproduce:** `python3 -I -B assemble.py MATH_REPO facts.json verdicts_verified.json OUT.json`
  regenerates the `records` array identically. `OUT.json` must not exist yet.
- **`singlepass_unverified/`** is the earlier single pass, kept for history. **Do not adopt it.**
