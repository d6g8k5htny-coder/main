# Record shapes: `q0.operations.registry/v1` and `q0.operation.trial/v1`

Both shapes are enforced by `tools/operations_check.py`; the vocabularies are
constants in `engine/operations/trial.py`, imported by the checker so the
writer and the checker cannot drift. Neither record carries any authority: the
registry's cells are the register's words, and a trial is a record of a
computation, not evidence.

## `REGISTRY.json` — `q0.operations.registry/v1`

| field | meaning |
|---|---|
| `schema` | `q0.operations.registry/v1` |
| `register` | where the cells come from, compared with the register the checker reads, never taken on trust: `path` (the `--register` file relative to the root), `tab` and `sheet_index` (the loaded tab's own), `export` (`trial.py`'s `REGISTER_EXPORT` verbatim) and `row_sha256_rule` (`ROW_SHA256_RULE` verbatim); exactly those five keys |
| `header` | the register tab's twelve column names, in order; must equal the register's header |
| `trial_ledger` | the ledger the checker reads, compared field by field: `path`, `tab`, `sheet_index` (the loaded `--trials-tab` file's), `columns` (exactly `trial.py`'s eighteen `TRIAL_COLUMNS`), `record_schema` (`q0.operation.trial/v1`) and `records_dir` (the `--trials` directory relative to the root); exactly those six keys |
| `authority` | exactly `trial.py`'s `REGISTRY_AUTHORITY`: the cells are transcribed, not decided; `git_side` is bookkeeping with no authority. Free text here once let "canonical and final" through; the checker now requires equality |
| `utility_and_novelty` | exactly `trial.py`'s `REGISTRY_UTILITY_AND_NOVELTY`: Utility and Novelty are the register's cells and never stronger (it carries `UNMEASURED` and `NOT_ASSESSED` verbatim) |
| `does_not_establish` | list of sentences (each ≥ 40 characters) that contains every sentence of `trial.py`'s `REGISTRY_DOES_NOT_ESTABLISH` verbatim; sentences may be added, the fixed ones may not be removed or replaced |
| `entries` | one per register row, no more, no fewer |

Every top-level string except `header` and the entries' cells is
repository-authored and is scanned for status words and usefulness words
exactly as `git_side` and a trial are (rules below).

Each entry has exactly these keys:

| key | meaning |
|---|---|
| `operation_id` | `OPnn`, the prefix of the `Operation` cell |
| `register_row_index` | 0-based index into the register tab's `rows`; each row transcribed once |
| `row_sha256` | `sha256(json.dumps(row, ensure_ascii=False, separators=(",",":")))` over the twelve-cell list |
| `cells` | the twelve cells verbatim under their header names, in header order — `Operation`, `Use when`, `Reuse state`, `Required scope`, `Action / output`, `Evidence checked`, `Exact source`, `Current review / authority`, `Do not infer`, `Next useful step`, `Utility`, `Novelty` |
| `git_side` | repository-side bookkeeping, kept separate; exactly the four keys below |

`git_side`:

| key | meaning |
|---|---|
| `exact_identity_checkable` | boolean; true only where the `Action / output` cell displays an identity exact rational or symbolic-exponent arithmetic reproduces in full from the cell alone |
| `trial_kind` | `EXACT_IDENTITY` or `NOT_MACHINE_CHECKABLE_HERE`; must agree with the boolean |
| `why` | a sentence saying why (for `EXACT_IDENTITY`, what precisely is reproduced and what is not) |
| `research_module` | `null`, or an existing path under `research/` holding related exact arithmetic |

`git_side` may not use a status word (`trial.py`'s `STATUS_WORDS`: the five
rule-1 families PROVE / PROVES / PROVED / PROVEN / PROOF, CERTIFY / CERTIFIES /
CERTIFIED / CERTIFICATE / CERTIFICATION, CLOSE / CLOSES / CLOSED / CLOSURE,
PROMOTE / PROMOTES / PROMOTED / PROMOTION, DISCHARGE / DISCHARGES / DISCHARGED,
and the review-verdict families INDEPENDENT, RATIFY / RATIFIES / RATIFIED /
RATIFICATION, VERIFY / VERIFIES / VERIFIED / VERIFICATION, ACCEPT / ACCEPTS /
ACCEPTED / ACCEPTANCE, ADMIT / ADMITS / ADMITTED / ADMISSION, PASS / PASSES /
PASSED / PASS_TECHNICAL, APPROVE / APPROVES / APPROVED / APPROVAL, SATISFY /
SATISFIES / SATISFIED, VALIDATE / VALIDATES / VALIDATED / VALIDATION, CONFIRM /
CONFIRMS / CONFIRMED / CONFIRMATION, ESTABLISHED, ESTABLISHES, and the
promotion / standing words CANONICAL, FINAL, AUTHORITATIVE, OFFICIAL, RELEASED,
SETTLED, HOLDS, RESOLVED; whole word, any case), and every sentence of it that uses a usefulness word (`USEFULNESS_WORDS`:
utility, novelty, useful, usefulness, novel, gain, valuable, beneficial,
benefit, improve, improvement, helpful, important, worthwhile, effective, ...)
must carry the register's `UNMEASURED` or `NOT_ASSESSED` verbatim and may not
pair the word with a value word or phrase (HIGH, LOW, MEDIUM, NEW, KNOWN,
MEASURED, ASSESSED, POSITIVE, SUBSTANTIAL, ..., BEYOND QUESTION, BEYOND DOUBT). Only the exact header phrase `Next
useful step` is exempt from that scan; the bare words `Utility` and `Novelty`
are not. Both scans are word lists applied to the text, not a reading of it: a
status or a usefulness claimed in words outside the lists is not caught, and
the `does_not_establish` sentences are what say the record moves nothing. An `EXACT_IDENTITY` entry must have an
identity spec in `trial.py` whose `displayed` text is a verbatim substring of
its `Action / output` cell and whose `scope_displayed` text, when present, is
a verbatim substring of its `Required scope` cell; a
`NOT_MACHINE_CHECKABLE_HERE` entry must have none.

## `trials/<Trial ID>.json` — `q0.operation.trial/v1`

Top-level keys, in this order and no others: the eighteen `operation_trials`
columns, then `does_not_establish`, then `authority`.

| key | type | rule |
|---|---|---|
| `Trial ID` | string | `TRIAL-<op>-<YYYYMMDDTHHMMSSZ>-<16 hex>`; the filename is `<Trial ID>.json`; unique; the stamp is `Run evidence.utc` and the hex is recomputed by the checker as `trial.py`'s `trial_id(op, Problem ID, Library digest, identity_spec, utc)`. Only records and `README.md` may exist in the directory; anything else is refused, and the README is scanned like a record |
| `Problem ID / SHA-256` | sha256 | the operation's `row_sha256` in the registry version named by `Library / catalog SHA-256`; anything else fails |
| `Split` | enum | `EXPOSED_DEVELOPMENT` for a run (the problem is the register's displayed cell); `NOT_APPLICABLE` for `NOT_RUN`; the checker refuses the other pairing |
| `Operation ID` | `OPnn` | must be in the registry |
| `Library / catalog SHA-256` | sha256 | of the bytes of `REGISTRY.json` at run time: the current file, or a version in git history (`git log --all -- engine/operations/REGISTRY.json`), in which case the record is HISTORICAL and every cell-bound check runs against that version; a digest matching neither fails |
| `Arm` | enum | `NONE_SINGLE_EXACT_REPLAY` for a run, `NOT_APPLICABLE` for `NOT_RUN`; no comparison arm exists here; the checker refuses the other pairing |
| `Budget / cost unit` | string | `trial.py`'s `BUDGET` verbatim: the unit (`checked_equalities`), the budget (none) and the failure rule; no other budget was run here |
| `Seed / environment` | object | exactly the keys `seed`, `python`, `implementation`, `platform`, `repository_commit_at_run`, `runner`, in that order; `seed` is `trial.py`'s `SEED` verbatim (none; deterministic), `runner` is `engine/operations/trial.py`, `repository_commit_at_run` is `null` or a 40-hex commit, `python` is a version string, the other two non-empty; a record cannot claim a seed, a runner or an interpreter the runner never had |
| `Verified result` | enum | `IDENTITY_REPRODUCED`, `IDENTITY_NOT_REPRODUCED`, `NOT_RUN` — closed. A `NOT_MACHINE_CHECKABLE_HERE` operation may only carry `NOT_RUN`, and an `EXACT_IDENTITY` operation may never carry it: the runner writes neither pairing, so a record with one was written by hand |
| `Search cost`, `Retrieval cost`, `Acquisition cost`, `Maintenance cost` | `NOT_MEASURED` | exactly that string: nothing here measures them, so a number there is a claim the runner never made and the checker refuses it |
| `Verification cost` | int ≥ 0 or `NOT_MEASURED` | the number of checked equalities for a run; exactly `NOT_MEASURED` for `NOT_RUN` (a number on an operation that was not run is refused) |
| `Timeout / failure charge` | int ≥ 0 | 0, or 1 when the computation raised; must be 0 for `IDENTITY_REPRODUCED` |
| `Run evidence` | object | `record_schema` = `q0.operation.trial/v1`; `trial_kind` (= the registry's for the operation); `identity_displayed` (= `identity_spec.displayed`); `reproduces` (= `identity_spec.reproduces`, a sentence saying what exactly is reproduced, or `null`); `identity_spec` (the spec run, bound to the cells; `null` for `NOT_RUN`); `arithmetic` (`trial.py`'s `ARITHMETIC` verbatim; no other arithmetic was run); `steps` — a list of `{statement, lhs, rhs, equal}` with exact sides rendered as strings or `{coefficient, exponents}` monomials; `checked_equalities` (= number of steps); `all_equal`; `failure` (`null` or `ExceptionName: message`); `utc` — **the time the runner ran**, from the clock at microsecond precision; never later than the checker's own clock and, for a record in `git HEAD`, never later than the commit that added it (the runner's `--utc` option is for tests only and a stamp from it has no place in a committed record); `reason_not_run` (the catalog's `git_side.why` verbatim for `NOT_RUN`; `null` for a run) |
| `Scope / constraints` | string | the register's `Required scope` cell verbatim |
| `Notes` | string | contains the register's `Do not infer` cell verbatim and the phrases `Utility: <cell>` and `Novelty: <cell>` verbatim |
| `does_not_establish` | list of sentences | each ≥ 40 characters; contains, verbatim, every sentence `trial.py`'s `required_does_not_establish(op, verdict)` returns — the four `GENERIC_DOES_NOT_ESTABLISH` sentences, the operation's `OP_DOES_NOT_ESTABLISH[op]` when one is defined, and `NOT_RUN_DOES_NOT_ESTABLISH` for `NOT_RUN`. Sentences may be added (they are scanned like any other authored text); the fixed ones may not be removed or replaced. This is the field that carries the weight against the word lists' gaps, which is why it is not free |
| `authority` | string | exactly `NONE — a trial is a record of a computation, not evidence` |

Consistency rules: for any run (`IDENTITY_REPRODUCED` or
`IDENTITY_NOT_REPRODUCED`) the recorded `identity_spec` must be displayed in
the named catalog version's cells and is re-run here; **the re-run's steps
must equal the recorded steps field for field** (`statement`, `lhs`, `rhs`,
`equal`), raise exactly when the record records a failure, and give the same
verdict. `IDENTITY_REPRODUCED` additionally requires at least one step, every
step `equal: true`, no `failure`, a zero failure charge, and that the spec is
the one `trial.py` binds to the operation. `IDENTITY_NOT_REPRODUCED` requires
a step that is not equal or a recorded failure. A run carries `Split`
`EXPOSED_DEVELOPMENT`, `Arm` `NONE_SINGLE_EXACT_REPLAY`, `Verification cost`
equal to the number of steps and `reason_not_run` `null`. `NOT_RUN` requires
no steps, no failure, no spec, `Split` and `Arm` `NOT_APPLICABLE`,
`Verification cost` `NOT_MEASURED`, and a `reason_not_run` equal to the named
catalog's `git_side.why` for the operation. Every rule in this paragraph is
enforced by the checker and has a negative control.

Time rules: `Run evidence.utc` may not be later than the checker's clock, and
a record present in `git HEAD` may not carry a `utc` later than the committer
time of the commit that added it (`git log --diff-filter=A`, compared at git's
one-second resolution; in a shallow clone the grafted root counts as the
adding commit, which only loosens the floor). The runner refuses a supplied
`--utc` later than its clock. A record whose stated time is false is exactly
the dishonest record this repository exists to exclude, and a round of
adversarial review found four such records (stamped 14:30:00Z, written at
13:11 UTC) about to be committed; they were replaced by records run from the
clock and the rules above were added.

No status word (`STATUS_WORDS` above; whole word, any case) may appear in any
repository-authored field, and every sentence of such a field that uses a
usefulness word must carry `UNMEASURED` or `NOT_ASSESSED` and no value word
(the rule stated for `git_side`, with the same word-list caveat). The scan covers every field, keys included,
except `Scope / constraints`, the quoted `Do not infer` cell inside `Notes`,
and any string that is exactly a register column name (such as the key
`Verified result`) — those are the register's words, not the trial's claim.
The writer applies the same scan and refuses to build a record that fails it.

No float appears anywhere in a trial record. Exponents are `Fraction`s
rendered as strings; constants such as `6^(2/3)/(3 kappa^(2/3))` are
`(rational coefficient, exponent vector)` monomials with integer bases reduced
to primes (`{"coefficient": "1/3", "exponents": {"2": "2/3", "3": "2/3", "kappa": "-2/3"}}`).

## What neither record establishes

A registry entry establishes that the register says what it says. A trial
establishes that a computation ran and what it computed. Neither is evidence
for a mathematical claim, neither measures utility or assesses novelty, and
neither moves a claim, premise, obligation, gate or grade.
