# `02.3_VALIDATION_RECEIPTS/01_VALIDATION_SUMMARIES_AND_RUN_BLOCKERS`

Drive folder id `1g2ddeJ87kC_Mk29a-R_fLVWygILe0SQY`. The 2026-09-17 inventory gives this
folder **5 items**, all native Google Docs, all held here as text exports — reading copies,
not the objects. The Drive titles are long and carry their own verdicts; they are preserved
unchanged in `dest`, with `/` written as the division slash `∕` because a filename cannot
contain a path separator.

## What is held, and at what exactness

| stored file | exactness | bytes on disk | Drive-reported size |
|---|---|---:|---:|
| `C047R-VAL-001-v1.0 — q-machine & verifier script validation receipt …` | reading copy | 3,170 | 1,969 |
| `C047R-VAL-002-v1.0 — Run-blockers: 20 scripts …` | reading copy | 2,561 | 3,899 |
| `C047R-VAL-003-v1.0 — Scripts that ran …` | reading copy | 3,554 | 2,177 |
| `C047R-VAL-004-v1.0 — Capsule ledger …` | reading copy | 3,606 | 1,862 |
| `C047R-VAL-006 — Disposition of the 13 deferred artifacts …` | reading copy | 2,835 | 1,861 |

## The status banners, verbatim

`C047R-VAL-001` states its own authority and its own scope limit:

> Authority: none.  Canonical impact: NONE.  Additive only. Companion to C047R-AUD-003.

> CONCLUSION: every q-machine / verifier script that is self-contained runs cleanly and emits its expected report; the rest are non-standalone by construction. No integrity or safety defect in the machine scripts (beyond the one cosmetic self-hash variant). Narrow scope: this confirms the scripts execute as published; it makes no claim about the q-machine's mathematical conclusions, P1.1 status, or any theorem.

It records `q0_verify.py` as

> 957,321 B materialized, py_compile OK; all embedded
> base64 batteries ran (self-contained). RUN OK.

(the file breaks the line after "all embedded"), and records four scripts as

> CORRECTLY REFUSED BY SAFETY POLICY (shell out via subprocess; benign build/validate drivers):

`C047R-VAL-002` opens

> C047R-VAL-002 — RUN-BLOCKERS: scripts that need co-located project files (not failures)

and says of each entry

> Each script below is correct but not standalone-runnable in isolation; to reproduce it, co-locate the named module/data file, then run with the venv python.

`C047R-VAL-004` is the capsule ledger. It records

> REAL BYTE-EXACT CAPSULES (5):

with two of them marked corrupt, one of those two also marked as having failed staging

> [CORRUPT] FAILED STAGING — GP-DATA-072-v1.0 Interval+Cone Source Cap

> [CORRUPT] GP-DATA-114-v1.0 — Byte-Exact Capsule for the Uniform-r Fi

and notes of the thirty-five documents it lists as not capsules

> "not-a-capsule" means only that THIS Doc had no BEGIN_SOURCE_GZIP_BASE64 block to byte-verify.

`C047R-VAL-006` explains why thirteen artifacts were not re-executed:

> These 13 could not be re-executed in this session for a MATERIALIZATION reason, not a math reason:

and concludes

> NET: of the 13, one (GP-DATA-035) is already independently confirmed; two (c020/c021 instr) are
> confirmed non-standalone library modules; the rest are integrity-re-check-pending or need their
> project bundle. None changes any conclusion of C047R-AUD-003. […]

(the file wraps that sentence across four lines; the ellipsis marks the close of the
entry, which continues with what a full closure would require)

## What this does not establish

These five documents are one line's summary of one line's session. They say so themselves:
additive, canonical impact NONE, authority none. **Nothing in them was re-run here**, and
`C047R-VAL-001`'s own narrow-scope sentence is the correct reading of the whole folder — a
script executing as published is not a statement about the mathematics, about P1.1's status,
or about any theorem. The capsule ledger's counts, including its two corrupt capsules, are
that session's findings and are neither confirmed nor contradicted by this port. All five
are reading copies; the Docs' bytes are not held, and no digest for them exists.
