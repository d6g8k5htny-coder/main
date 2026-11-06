# CBZ proof and quarantine index

**State: incomplete inventory; CBZ-0 is not established.**

This is the Main cold-start map for proof status. It records exact boundaries;
it does not turn a source census, empirical check, review, or successful CI job
into a kernel proof. A theorem is CLOSED only for the literal statement and
allowed-axiom boundary actually checked. Anything else must remain OPEN or be
placed in a complete dormant quarantine.

## Snapshot binding

| Repository | Bound revision | Role |
|---|---|---|
| Main | Candidate containing this file; native source parent `e4c2518a8b76ad884c847faddcaa6bee916ee666`, based on default `299c7fad72f1079b87b2d044b2d30f2098678f29` | This index and Main formal/tool sources; no proof status changed by the G11 repair |
| Math- | `f532c3e2d91a0c7679a37c733741660917a9b21b` | Repository-local mathematical proof catalogue and retained evidence |
| Universal-Law-Workspace | `05f3096d077a6fa4ff6cf763c074451cb1ac9972` | Federation pin snapshot; its Main pin is the preceding default, not this later candidate |

The observations are sequential and can become stale. [BASELINE.md](BASELINE.md)
records the current criterion and gate boundary.

## What is presently indexed

- The current Math- repository has its own
  [proof availability index](https://github.com/d6g8k5htny-coder/Math-/blob/f532c3e2d91a0c7679a37c733741660917a9b21b/PROOF_INDEX.md).
  That file and its linked artifacts retain their own dates, source cuts,
  statements, axioms, outcomes, failures and limitations. They are not
  promoted here to a whole-federation closure claim.
- Main retains formal sources under `formal/`, the checker
  `tools/formal_gate_check.py`, and the required-check contract in
  [docs/FORMAL_REQUIRED_CHECKS.md](docs/FORMAL_REQUIRED_CHECKS.md). A fresh
  current-composition run is still required. This index does not add a CLOSED
  disposition for any Main declaration merely because the files exist.
- The closed-unmerged foundation handoff in
  [Main #695](https://github.com/d6g8k5htny-coder/main/pull/695), head
  `82d9e480a51b946b3e74525ccf32139129d37d28`, remains a recoverable candidate
  record. It is not the default baseline and is not silently rebound to this
  revision.
- Empirical statement results from `tools/cbz/statement_fuzz.py` are bounded
  PASS, CORROBORATED, FAIL or UNSUPPORTED observations. They are never kernel
  evidence, proof closure, scientific alignment or scientific acceptance.

## Unclosed federation boundary

The following whole-federation work is not yet proved complete:

1. a machine-readable declaration-by-declaration reconciliation of every
   current Main and Math- proof item;
2. an exact CLOSED/kernel receipt or a complete dormant quarantine for every
   item;
3. a fresh current-composition axiom, Lean build and kernel pass;
4. a verified absence of `sorry`, `admit`, unchecked axioms and presented
   theorem stubs across the full admitted scope;
5. proof that closed code has no dependency on quarantined logical content;
6. one fresh final G0–G11 pass over the same bound snapshot.

Until all six are observed, criterion 3 and the proof-related parts of
criteria 4 and 6 remain unawarded.

## Dormant quarantine contract

A dormant record is complete only when it carries all eleven required fields:

1. id;
2. title;
3. exact location;
4. Layer A status;
5. exact proven and unproven boundary;
6. mathematical blocking reason;
7. attempted work;
8. resumption cost;
9. isolation;
10. `owner: none`;
11. `maintenance: zero`.

Missing procedural review is not itself a mathematical defect. A quarantine
may preserve an unresolved statement, failed attempt or historical delivery,
but admitted closed code must not depend on its logical content. Quarantine
does not refute the work, erase authorship, or change a historical outcome.

## Reading rule

Use the exact native commit, statement text, hypotheses, axiom report and
execution receipt together. Do not infer a continuum result from finite
fixtures, scientific acceptance from a kernel check, current qualification
from cached evidence, or a stronger theorem from a checked conditional.
