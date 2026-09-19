# `engine/operations/trials/` — trial records, append-only

One `q0.operation.trial/v1` record per run of `engine/operations/trial.py`,
named `<Trial ID>.json`, in the shape of the register's `operation_trials`
tab (its eighteen columns as top-level keys, plus `does_not_establish` and
`authority`). The shape is in `../SCHEMA.md`. Only these records and this
README may exist here; `tools/operations_check.py` refuses any other file,
and it applies to this README the same two word-list scans (status words;
value words next to the register's UNMEASURED / NOT_ASSESSED markers) it
applies to a record.

Records here are **append-only**: the writer refuses an existing path, and
the checker fails if a record present in `git HEAD` differs from, or is
missing from, the working tree. A record whose catalog digest is that of an
earlier committed `REGISTRY.json` is reported as HISTORICAL, kept, and
checked in full against that version; a record naming a catalog version
nobody can read fails. Every record's recorded arithmetic is re-run by the
checker and must match step for step.

**A record's `Run evidence.utc` is the time the runner ran**, read from the
clock at microsecond precision. The runner's `--utc` option exists for the
tests only; a committed record never carries a supplied stamp. The checker
refuses a record whose `utc` is later than its own clock and, for a record
in `git HEAD`, later than the commit that added it.

**A trial is a record of a computation, not evidence.** `IDENTITY_REPRODUCED`
says that the algebra the register's `Action / output` cell displays was
reproduced in exact arithmetic from the cell alone. It does not say the
operation's premises hold anywhere, that any caller meets its `Required
scope`, or that the cited source is correct. Utility is `UNMEASURED` and
Novelty is `NOT_ASSESSED`, which are the register's words; no record here
measures the one or assesses the other. No record here moves a claim,
premise, obligation, gate or grade, and none earns independence credit for
anything.
