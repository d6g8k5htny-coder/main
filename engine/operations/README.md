# `engine/operations/` — the register's reusable operations, worked actively, with trials that are records

The additive [RN applicability candidate](../../docs/RN_AFFINE_MOMENTS.md)
now exposes exact correlated Gaussian moment polynomials through
`rn_applicability.py`. It binds source bytes, dimension, coordinate order,
normalization, conditioned law, domain and evidence tier before computing a
whole-mark bound. Its separate finite cost experiment compares polynomial
reuse with recomputation. It does not alter this catalog's transcribed
Utility/Novelty cells or reinterpret existing trial records; canonical
utility remains UNMEASURED and novelty NOT_ASSESSED. The mathematical result
is an unreviewed algebra candidate, with zero independence credit.

The 2026-09-18 xlsx export of GP-REG-032-v1.2 added two tabs to the coupled
registers: `reusable_operations` (sheet 42; fifteen rows OP01–OP15, twelve
columns) and `operation_trials` (sheet 43; an eighteen-column header and no
rows). The Drive's own guide for them is mirrored byte-exact under
`drive/deltas/2026-09-18/2026-09-18_REUSABLE_OPERATIONS/`; its words are data,
and the ones that matter here are: *"Operation Trials is intentionally empty"*,
*"All have UNMEASURED usefulness and NOT_ASSESSED novelty"*, and *"A name,
reading copy, approval or self-citation earns no usefulness credit."*

This directory lets the repository work those operations actively — run an
operation's **displayed** exact identity in exact arithmetic and record the
run in the register's own trial-ledger shape — without any trial ever becoming
evidence or moving a status.

> **A trial is a record of a computation, not evidence.** `IDENTITY_REPRODUCED`
> means the algebra the `Action / output` cell displays was reproduced from the
> cell alone over `fractions.Fraction`; it does not mean the operation's
> premises hold anywhere, that any caller satisfies its `Required scope`, that
> the source the cell cites is correct, or that the operation is useful or new.
> **Utility is `UNMEASURED` and Novelty is `NOT_ASSESSED` because the register
> says so**; this directory transcribes those words and never says more, and
> `tools/operations_check.py` fails if it does. Nothing here moves a claim,
> premise, obligation, gate or grade. The five validity premises of Theorem D1
> v2.2(2) are OPEN and `D3-LEMMA-RN-UNIF` remains open.

## What is here

```
REGISTRY.json   q0.operations.registry/v1 — one entry per register row, the twelve cells
                verbatim under their header names, the row's canonical digest, and a
                separate git_side block saying what this repository can compute
trial.py        the runner: exact monomial / rational arithmetic, the four identity specs
                bound to their cells, the append-only writer, the CLI
trials/         q0.operation.trial/v1 records, one file per trial, append-only
SCHEMA.md       the two record shapes, field by field
```

```bash
python3 engine/operations/trial.py --list                                   # the fifteen, with their trial kind
python3 engine/operations/trial.py --op OP02 --out engine/operations/trials/
python3 engine/operations/trial.py --all  --out engine/operations/trials/   # every EXACT_IDENTITY op
python3 tools/operations_check.py                                           # non-zero on any problem
python3 -m pytest -q tests/test_operations.py                               # negative controls
```

The runner exits 0 when the record was written and the verdict is
`IDENTITY_REPRODUCED` or `NOT_RUN`, 1 when it is `IDENTITY_NOT_REPRODUCED` (the
record is still written: a failed reproduction is a record too), and 2 when it
refused to write (unknown operation, identity not displayed in the cell,
destination exists or is governed, or a supplied timestamp later than the
clock).

**A record's `Run evidence.utc` is the time the runner ran**, read from the
clock at microsecond precision. The runner's `--utc` option exists for the
tests only (two runs are made to collide, and sandbox records get a known
stamp); a committed record never carries a supplied stamp. The checker refuses
a record dated later than its own clock and a committed record dated later
than the commit that added it. An adversarial round found the first four
records of this directory stamped `14:30:00Z` through `--utc`, seventy-eight
minutes after they were written; they were withdrawn before commit and
replaced by records run from the clock.

## Which operations are checkable here, and what exactly is reproduced

`exact_identity_checkable` is true only where the `Action / output` cell
displays an identity that exact rational or exact symbolic-exponent arithmetic
can reproduce **in full from the cell alone**. Four of the fifteen qualify:

| op | what the trial reproduces, exactly | what it does not touch |
|---|---|---|
| OP02 | the exponent ledger `(r^3/6) r^-5 r^2 = 1/6`: exponents 3 − 5 + 2 = 0, coefficients multiply to 1/6, the product is the constant 1/6 | whether the `r^2` factor is justified (the register: "must be established in the caller"); 1/6 is not a contact coefficient |
| OP03 | with the gap map `ell = kappa r^3/6` from `Required scope`: the inverse `r = (6 ell/kappa)^(1/3)` returns `ell`; `r · dr/dell = 6^(2/3)/(3 kappa^(2/3)) · ell^(-1/3)`, with the constant held as the monomial `1/3 · 2^(2/3) · 3^(2/3) · kappa^(-2/3)` and never as a float | the input measure `r dr` (a caller premise); any asymptotic; weighted mass, tails, mark integration, selection bridge |
| OP04 | **exponent arithmetic only**: the displayed exponents `1/4, 1/4, 1/2` sum to one and are the reciprocals of the `Required scope` moment orders `4,4,2` (two checked equalities; the record's `Run evidence.reproduces` says so in words) | the inequality `E[|ABC| I] <= (E A^4 E B^4)^(1/4) (E C^2)^(1/2)` itself, which is a theorem about expectations, not an identity; the finiteness of any moment; any SIDE24 input |
| OP05 | the displayed witness only: `A = B = 1, C = 1/4` give `LHS = 1/4`, the wrong `RHS = (E C^4)^(1/2) = 1/16` (exact integer roots), and `1/4 > 1/16` (three checked equalities; the record's `Run evidence.reproduces` says so in words) | anything beyond that one point; the corrected OP04 bound `(E C^2)^(1/2)` is not evaluated at the witness because the cell does not display it; RN5's separate Gaussian witness is not replayed |

The other eleven are `NOT_MACHINE_CHECKABLE_HERE`, each with its reason in
`git_side.why`: OP01 refers to a stored frame the cell does not display; OP06
names `a`, `q_adj`, `eta` without defining them in the cell; OP07, OP09, OP10
and OP11 are universal inequalities or transcendental expressions, which
finitely many exact evaluations do not reproduce; OP08 is a trigonometric
integral over an undisplayed covariance; OP12 is a workflow; OP13 a logical
boundary; OP14 a review route; OP15 an authority boundary. **That
classification is a statement about this repository's reach, not about the
operation**: false does not mean wrong, unverifiable or unverified elsewhere.

Every identity spec in `trial.py` names, in `displayed`, the exact substring of
the register's `Action / output` cell it reproduces, and, where it also takes
data from `Required scope` (OP02's three factors, OP03's gap map, OP04's
moment orders), that substring in `scope_displayed`. The runner and the checker
both refuse a spec whose displayed text is not in its cell, so the arithmetic
cannot drift from the register's words. Every entry carries the register's
`Do not infer` cell verbatim, and every trial repeats it verbatim in `Notes`.

## How the `operation_trials` columns are filled

The trial record's top-level keys are exactly the register tab's eighteen
columns, in order, plus `does_not_establish` and `authority`. Honest values:

| column | value here |
|---|---|
| Trial ID | `TRIAL-<op>-<utc>-<16 hex of the input digest>`; deterministic from the operation, the row digest, the registry digest, the identity spec and the timestamp, and recomputed from the record by the checker |
| Problem ID / SHA-256 | the entry's `row_sha256` (sha256 of `json.dumps(row, ensure_ascii=False, separators=(",",":"))`) in the registry version the record names |
| Split | `EXPOSED_DEVELOPMENT` for a run — the problem is the register's own displayed cell, which the Drive guide itself calls an exposed development test, not a held-out usefulness trial; `NOT_APPLICABLE` for `NOT_RUN`; the checker refuses the other pairing |
| Operation ID | `OPnn` |
| Library / catalog SHA-256 | sha256 of the bytes of `REGISTRY.json` the trial ran against: the current file, or a version of it in git history (then the record is HISTORICAL and checked against that version); any other digest fails |
| Arm | `NONE_SINGLE_EXACT_REPLAY` for a run — no comparison arm (fixed library / lemma cache / scope-aware operations) is run here; `NOT_APPLICABLE` for `NOT_RUN`; the checker refuses the other pairing |
| Budget / cost unit | `trial.py`'s fixed sentence, verbatim: `unit=checked_equalities`; the budget is none; the failure rule is stated |
| Seed / environment | exactly the runner's six keys: `seed` (the fixed sentence: none, deterministic), `python`, `implementation`, `platform`, `repository_commit_at_run` (`null` or a 40-hex commit), `runner` (`engine/operations/trial.py`); the checker refuses any other shape or value |
| Verified result | closed vocabulary: `IDENTITY_REPRODUCED`, `IDENTITY_NOT_REPRODUCED`, `NOT_RUN`; a non-checkable operation carries only `NOT_RUN`, and a checkable one never does |
| Search / Retrieval / Acquisition / Maintenance cost | exactly `NOT_MEASURED` — nothing here measures them, so nothing here claims them, and the checker refuses a number |
| Verification cost | the number of equalities checked, an integer of the stated unit; exactly `NOT_MEASURED` for `NOT_RUN` |
| Timeout / failure charge | `0`, or `1` when the computation raised (then the result is `IDENTITY_NOT_REPRODUCED`; a raise is never swallowed into success) |
| Run evidence | the arithmetic performed, as data: the spec, what it reproduces (`reproduces`, where the spec says so), every step with its exact left and right side and whether they were equal, the failure if any, the fixed `arithmetic` sentence, and for `NOT_RUN` the registry's `git_side.why` verbatim as `reason_not_run`; the checker re-runs the spec and requires the same steps field for field |
| Scope / constraints | the register's `Required scope` cell verbatim |
| Notes | the register's `Do not infer` cell verbatim, the trial kind, and the register's Utility / Novelty cells as `Utility: <cell>; Novelty: <cell>` verbatim |
| does_not_establish | the runner's fixed sentences, verbatim: the four generic ones, the operation's own, and the `NOT_RUN` sentence for `NOT_RUN`; the checker requires every one of them and allows (and scans) additions. This is the field that carries the weight against the word lists' gaps, so it is not free text |

Records are append-only: the writer uses `open(path, "x")` and refuses an
existing path, and `tools/operations_check.py` compares every committed record
against `git HEAD` byte for byte. A record whose catalog digest is that of an
earlier committed `REGISTRY.json` is reported as HISTORICAL, kept, and checked
in full against that version's cells (resolved through `git log --all`); a
record whose digest matches no readable version fails. Neither is evidence of
anything. Because the checker re-runs every record's spec and demands the same
steps, a change to a spec or an evaluator in `trial.py` makes the existing
records of that operation fail; that is the intended signal, and no remedy is
defined here.

## What the checker enforces

`tools/operations_check.py` exits non-zero unless every registry entry equals
its register row cell for cell with the right digest and there are exactly the
register's rows; Utility and Novelty are the register's cells, and neither
`git_side` nor the registry's own `authority`, `utility_and_novelty` and
`does_not_establish` text uses a status word or speaks of utility, novelty,
usefulness or gain except in a sentence that carries the register's
`UNMEASURED` / `NOT_ASSESSED` and pairs the word with no value word; every
`EXACT_IDENTITY` entry has a spec bound to its cells and no other entry has
one; the registry's `authority`, `utility_and_novelty` and
`does_not_establish` are `trial.py`'s fixed sentences (equality for the first
two, containment of every fixed sentence for the third) and its `register`
and `trial_ledger` provenance blocks describe the files the checker was
actually pointed at (tab, sheet index, path, columns, record schema, records
directory); every trial has exactly the twenty keys, names a registered
operation and a readable catalog version (current, or in git history) whose
row digest it carries, carries a result from the closed vocabulary, carries
the `Do not
infer` cell and the `Utility: <cell>` / `Novelty: <cell>` phrases verbatim in
`Notes` and the `Required scope` cell verbatim in `Scope / constraints`, uses
none of the words in `trial.py`'s `STATUS_WORDS` in any repository-authored
field (keys included) — the five rule-1 families PROVE / PROOF, CERTIFY /
CERTIFICATE, CLOSE / CLOSURE, PROMOTE / PROMOTION and DISCHARGE in participle,
verb and noun forms, and the review-verdict families INDEPENDENT, RATIFY,
VERIFY, ACCEPT, ADMIT, PASS, APPROVE, SATISFY, VALIDATE, CONFIRM, plus
ESTABLISHED / ESTABLISHES and the promotion / standing words CANONICAL, FINAL,
AUTHORITATIVE, OFFICIAL, RELEASED, SETTLED, HOLDS, RESOLVED — and uses none of
the words in `USEFULNESS_WORDS`
(utility, novelty, useful, gain, valuable, beneficial, improve, important,
...) outside a sentence that carries the register's `UNMEASURED` /
`NOT_ASSESSED` and pairs the word with no value word; **both are word lists,
not a reading of the sentence**, so a status or a usefulness claimed in other
words is not caught, and the record's `does_not_establish` is what says it
moves nothing. Every trial is internally consistent (every step equal for
`IDENTITY_REPRODUCED`; `trial_kind`, `identity_displayed`,
`checked_equalities` and the Trial ID digest recompute from the record;
`Split`, `Arm` and `Verification cost` agree with the verdict; the four
unmeasured costs are exactly `NOT_MEASURED`; `Budget / cost unit` and
`Run evidence.arithmetic` are the runner's fixed sentences; a `NOT_RUN`
record's `reason_not_run` is the registry's `git_side.why` verbatim; `Seed /
environment` is the runner's six keys with its fixed `seed` and `runner`
sentences; `does_not_establish` carries every fixed sentence the runner
writes), has its recorded spec re-run here with **the same steps field for
field** and the same verdict, is `NOT_RUN` with no spec when its operation is
not checkable and never `NOT_RUN` when it is, carries a `Run evidence.utc` no
later than the checker's clock and, once committed, no later than the commit
that added it, and is unchanged since `git HEAD`. Under `trials/` only records
and `README.md` may exist, and the README is scanned like a record. The runner holds itself to the same
status-word and usefulness-word rule and refuses to write a record that
breaks it. `tests/test_operations.py` breaks copies in each of those ways
(including every mutation three adversarial rounds found slipping through:
fabricated step values, review-verdict words and their verb and noun forms,
usefulness synonyms, `Utility HIGH` in a trial or in the registry's own text,
a made-up catalog digest, a stale row digest under the current catalog,
rewritten `Run evidence` bookkeeping, a zeroed Trial ID digest, a `NOT_RUN`
record dressed with a verification cost, a development split and a replay
arm, a numbered search cost, a rewritten arithmetic sentence, a paraphrased
`reason_not_run`, a record dated 2099, a record committed with a date before
its run, a bland one-sentence `does_not_establish` in a trial and in the
registry, a registry authority reading "canonical and final", provenance
blocks naming another tab and a two-column ledger, a hand-written `NOT_RUN`
for a checkable operation, a seed of 42 under Python 2.7, and a `SUMMARY.md`
claiming PROVEN dropped into `trials/`) and asserts the refusal through the
checker's CLI, and proves that a deliberately wrong identity yields
`IDENTITY_NOT_REPRODUCED`, that an overwrite is refused and that a future
timestamp is refused.

## What this directory does not establish

Nothing mathematical. It does not establish that any operation is correct,
useful, new, reviewed or admissible anywhere — the register's `Reuse state`,
`Current review / authority` and `Do not infer` cells govern and are carried
verbatim. It does not measure utility (no held-out problems, no matched
budgets, no comparison arms, no paired cost `G_n(R)`; the Drive guide's
protocol for that is not run here) and it assesses no novelty. A trial that
reproduces an identity is a record that a computation ran; it is not evidence,
not a review verdict, not a certificate and not a status, and it earns no
independence credit for anything. It does not compose the 2D upper or lower
tracks with the 3D lifetime track. No original prize problem is solved, and the
prize track is not touched.
