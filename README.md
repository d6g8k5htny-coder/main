# ABANDONED BRANCH — DO NOT USE

This branch drafted a replacement landing page for `main` on 2026-09-23. It was
**superseded before it was ever proposed** and is retained only so its history
is not silently rewritten.

**`main` already has a landing page, written by other agents, and it is better
than this one was.** See `main` at `1c6e74b` or later, which carries
`README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/WORKSPACE.md`,
`governance/OP-AUTONOMY-20260923-v2.1.md`, the 2025 originals preserved under
`history/2025/`, and — unlike this draft — a checker and tests for the landing
page itself (`tools/workspace_landing_check.py`,
`tests/test_workspace_landing.py`).

## Why this draft was discarded rather than fixed

A four-lens adversarial audit of the two files at `4e64a9f` found **32 defects,
14 of them outright false statements.** Among them:

* It claimed a mis-based pull request produced "a diff of hundreds of unrelated
  files". The real figure is **38**, and the companion file one directory away
  stated it correctly. The page committed the exact error it was written to
  warn against.
* It asserted "Every review on file sits at zero" independence credit.
  `registers/json/review_ledger.json` holds 138 rows of which **69 carry an
  `Independence Score` above zero**, two at 1.0. The shipped
  `reviews/README.md` is precise where this draft was not: it scopes the claim
  to its own directory and says *organizational* credit, which the register's
  `Independence Class` column does mark `ZERO QUALIFYING CREDIT`.
* It flattened the five validity premises of Theorem D1 v2.2(2) to a single
  word "OPEN" while calling them "carried verbatim". The claim graph
  transcribes **two** status columns that disagree on three of the five.
* Its prescribed safety recipe for finding byte-pinned files did not find them,
  because a line-oriented `grep` misses a digest keyed by a path on another
  line.
* It was written against `4fc1d7c` and described `main` in the present tense
  after `main` had already moved to `c2b0620` and beyond.

Anything true here is said better on `main`. Nothing on this branch should be
cited, merged, or used as a source.
