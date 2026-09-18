# Exception recovery

The 2026-09-17 accessibility publication retained **137 exceptions** and invented
nothing to fill them. This directory is an attempt to recover the bytes behind
the 31 exceptions in the four classes named in
[`docs/CONTRIBUTION_PLAN.md` §7](../docs/CONTRIBUTION_PLAN.md) — 8
`EMPTY_NATIVE_BODY`, 5 `READ_FAILED`, 3 `ARCHIVE_READ_FAILURE`,
15 `ENCODED_BLOCK_FAILURE` — and a record of exactly how far each attempt got.

The counts were re-enumerated from `drive/source_map/Exceptions.csv` rather than
carried over: 137 rows, 8 / 5 / 3 / 15 in those four classes. They agree with
`docs/OPEN_PROBLEMS.md` §E. (One cross-source difference outside these classes is
noted in the ledger: the inventory carries `BINARY_UNRENDERED` on two objects
where the exceptions file carries one row. Reported, not repaired.)

## What is here

```
LEDGER.json    one record per exception: class, object, every route attempted,
               outcome, digest corroboration, and — where nothing was recovered —
               exactly what is missing
recovered/     bytes whose SHA-256 matches a digest the corpus itself states
candidates/    bytes that are probably the right content and demonstrably not
               the right bytes
```

Every blob carries its own digest in its filename
(`<name>.<sha256>.bin`). `tools/recovery_check.py` recomputes it.

## Result

| Outcome | Count |
|---|---:|
| RECOVERED (digest corroborated) | 13 |
| CANDIDATE (digest **not** corroborated) | 3 |
| UNRECOVERABLE | 15 |

The distinction between the first two rows is the point of the whole exercise. A
recovery is bytes that reproduce a digest the corpus states. Everything else is a
candidate, however plausible it looks, and candidates live in a separate
directory so that no consumer can mistake one for a source of record.

### The named priority was not recovered

`LS-DATA-015-v1.0-R1` and `-R2`, the byte-exact and hex-gzip TB-G2 algebra result
capsules, are **UNRECOVERABLE**. All four routes were tried and all four are
closed: no archive member carries the payload, no mirror carries it, a fresh
2026-09-18 download returns a body consisting of nothing but a UTF-8 BOM, and
Drive reports `createdTime == modifiedTime` for each shell — so there is no
earlier revision holding a body. The malformed `LS-DATA-015-v1.0` predecessor
those capsules were created to correct is itself truncated (its two encoded
blocks are 176 and 17 base64 characters short of their declared lengths), so
there is nothing to regenerate from either.

That is the honest answer and it is a real finding: the capsules whose entire
purpose was to carry exact bytes were **created empty and never filled**. The
TB-G2 dependency is missing at the source, not merely unread.

### What did come back

Eight of the ten `LS-DATA-009-v1.1` chunk exceptions turned out to be an
**extraction-rule artifact**, not damaged data: each block begins with its own
`Chunk characters:` / `Chunk SHA-256:` metadata lines, which a naive
BEGIN-to-END extractor swallows into the payload. Skipping them makes the blocks
decode and reproduce their own declared digests, and the carrier's own
reconstruction rule then yields the declared 10,953-byte result with the declared
SHA-256. The two blocks that still fail are the two the carrier itself marks
`FAILED / DO NOT USE`, and this pass measured their defects independently —
including the two missing characters `zM` the carrier's erratum names.

Three of the five `READ_FAILED` members and one `ARCHIVE_READ_FAILURE` were
recovered byte-exactly from mirror carriers and published reading copies. The
ledger's `findings` section records four further byte-level observations,
including a case where the published reading copy and the carrier's own declared
identity are **different bytes** for the same declared artifact.

Those findings are offered as facts about bytes. They are not verdicts.

## What recovery does **not** do

This is the discipline the audit itself applied, restated for this directory.

* **Restoring bytes is not review.** Nothing here reviews, validates, checks or
  endorses any restored object. A recovered file has been *read*, not *assessed*.
* **Recovery promotes nothing.** No claim, premise, obligation or register status
  is promoted, closed, discharged or reclassified by anything in this directory.
  A recovered capsule enters the corpus **at the status its register row already
  carries — not higher**. If a row says `NEEDS_RECONCILIATION`, it still says
  `NEEDS_RECONCILIATION` with the bytes in hand.
* **Digest corroboration is not correctness.** It establishes that these bytes
  are the bytes some corpus record names. It does not establish that the naming
  record is right, that the content is mathematically sound, or that a
  certificate inside it holds.
* **A CANDIDATE is not a recovery.** Candidates are stored apart, are never
  described as byte-exact, and must not be consumed as a source of record while
  their digests stand uncorroborated.
* **No independence credit is created here, at all.** This session is
  Anthropic-family. Under `OP-PROT-019-v1.1` (R17) §4 organizational
  independence is recorded separately and, for a same-provider reviewer, at
  **zero** — and this work is not even review, so it earns nothing on any axis.
  Every independence-requiring gate that touches these objects **remains open**,
  and would remain open even if every record here said RECOVERED. Where a carrier
  is OpenAI-authored rather than Anthropic-authored, the ledger says so and still
  records the credit the predicate actually licenses: none.
* **An UNRECOVERABLE record is the correct output, not a failure.** Fifteen of
  the thirty-one records say so, each naming exactly what is absent — byte
  counts, digests, character deficits. Naming the hole precisely is what lets
  someone else fill it. Inventing content to close it would destroy the one
  property that makes this corpus worth anything.
* **Nothing here composes the 2D upper or lower tracks with the 3D lifetime
  track, and no original prize problem is solved.**

## Checking it

```bash
python3 tools/recovery_check.py     # custody invariants; exits nonzero on failure
python3 -m pytest -q tests/test_recovery.py
```

`tools/recovery_check.py` asserts that every stored blob's digest matches its
path, that no candidate is stored among the recoveries, that every ledger record
names at least one attempted route, that an `UNRECOVERABLE` record names what is
missing, that no recovered payload appears in
[`quarantine/EXCLUSIONS.json`](../quarantine/EXCLUSIONS.json), and that the
independence credit in the ledger is 0. `tests/test_recovery.py` carries the
negative controls: a corrupted blob, a candidate promoted into `recovered/`
without corroboration, a candidate merely relabelled, a quarantined digest, a
record with no attempted route, and a non-zero independence credit must each make
the checker fail.
