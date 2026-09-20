# `01_ACTIVE_RESEARCH_PACKAGES/12_P1.1_LAW_SPECIFIC_Q_MACHINE`

Drive folder id `14ATuEpywQvsv9cteXfZ_lpv7RjilcJaw`. The 2026-09-17 inventory gives this
lane **145 items** — 116 files and 29 folders — and every one of them appears exactly once
across the 24 `_MANIFEST.jsonl` files under this directory. Nothing in the lane is omitted
and nothing is listed twice; `tools/verify_manifests.py` recomputes every digest and byte
count here, and `tools/mirror_quotes_check.py` checks every quotation below against the
stored bytes.

**A mirror is a copy. It is not review, replay, endorsement, promotion or acceptance, and
it moves no status.** Every status word in this file is transcribed from a source and
attributed to it. None is decided here.

## What is here

| | items | bytes on disk |
|---|---:|---:|
| byte-exact against the 2026-09-17 inventory digest (`exact: true`) | 91 | 8,529,752 |
| text exports of native Google Docs — reading copies (`exact: false`) | 25 | 114,591 |
| tree-only rows (all 29 of them folders) | 29 | 0 |
| **total** | **145** | **8,644,343** |

The 91 byte-exact files are 62 `application/json`, 18 `text/x-python-script` and 11
`text/markdown`. Each was downloaded through the Drive connector on 2026-09-20, decoded
from this session's transcript on disk, then hashed and byte-counted from disk; a row is
`exact: true` only where **both** the SHA-256 and the byte count equal what
`drive/inventory.jsonl` declares for that Drive id. All 91 matched. No digest disagreed,
and nothing in this lane failed to fetch.

The 25 native Google Docs carry no payload digest anywhere in the corpus, so their
`inventory_sha256` is `null`, the `sha256` recorded for them is the export's own —
computed here for the first time — and their `inventory_bytes` is the size Drive reports
for the Doc, which is not a digest of anything and is not what the stored export weighs.
Google Docs normalises formatting on export. **A reading copy is not the object.**

A note on a field that reads oddly: the inventory's `access_status` for 90 of the 91
byte-exact files is `TEXT_READING_COPY`, and for `q0_machine.json` it is
`EXACT_TOKEN_TABLE`. Those labels are the 2026-09-17 accessibility publication describing
how *it* read the object; they are transcribed into the manifest rows unchanged. They say
nothing about the bytes stored here, whose identity rests on the recomputed digest.

## Two things this port contradicts, and does not fix

**1. `engine/carriers/MANIFEST.json`, carrier `CR-Q0-VERIFY`.** That record describes
`q0_verify.py` (Drive id `1FxdPkPmK-WhTm-9hQwyJsuJiscCJoy-7`, 957,321 B) as not read,
recorded from the inventory only, with `blob_stored: false`, `blob_path: null`,
`not_stored_reason: SIZE`, `arithmetic: unknown`, and a note saying that because it was
not downloaded nothing about its arithmetic is known there. **That is no longer true of
this repository.** The file has now been downloaded, its digest recomputed and found equal
to the inventory's, and its bytes stored at
`02.1_MACHINE_ROOT_SCHEMA/01_PRIMARY_MACHINE_ROOT/q0_verify.py`. The carrier record and
this lane therefore disagree about whether the repository holds those bytes. The
5,910,703-byte `q0_machine.json` beside it is a different case and not a contradiction:
`engine/carriers/` has no record for it at all — its Drive id
`1PXa-ZqCrICicUUDIy37PafHgdjCOb3cj` appears nowhere under `engine/` and neither does the
string `q0_machine` — so there is nothing for this lane to disagree with. This port does not touch
`engine/carriers/`; the disagreement is reported for the integrator to resolve, and until
it is resolved the carrier record is the stale one. Storing the bytes changes nothing else
about that carrier: it is still not certifying, still `lane: D`, and this lane neither read
its arithmetic nor ran it.

**2. `recovery/LEDGER.json`, exceptions ENB-01 through ENB-04-S1.** The ledger records the
four TB-G2 / LS-DATA capsule documents at this lane's root as UNRECOVERABLE with
`stored_path: null`, and names them the priority recovery item. This port stores a text
export of each. Each export is **exactly three bytes** — the UTF-8 byte-order mark
`EF BB BF` — and nothing else: no capsule payload, no Base64 block, no source. **Storing
three bytes of byte-order mark is not a recovery, and it does not change the ledger's
finding.** It corroborates it: what route (c) of that ledger found by reading is what is
now held on disk. This port does not touch `recovery/`.

The four items at this lane root are exactly those documents:

| stored file | exactness | bytes on disk | Drive-reported size | access_status |
|---|---|---:|---:|---|
| `LS-DATA-013-v1.0 — Byte-Exact Gzip Capsule Package — q0 Verifier 1.2.1 Hardening.export.txt` | reading copy | 3 | 1,024 | `EMPTY_NATIVE_BODY` |
| `LS-DATA-015-v1.0-R1 — Corrected Byte-Exact TB-G2 Algebra Companion Capsule.export.txt` | reading copy | 3 | 1,024 | `EMPTY_NATIVE_BODY` |
| `LS-DATA-015-v1.0-R1 — Corrected Byte-Exact TB-G2 Algebra Result Capsule.export.txt` | reading copy | 3 | 1,024 | `EMPTY_NATIVE_BODY` |
| `LS-DATA-015-v1.0-R2 — Corrected Hex-Gzip TB-G2 Algebra Result Capsule.export.txt` | reading copy | 3 | 1,024 | `EMPTY_NATIVE_BODY` |

All four export to identical bytes and therefore to one digest,
`f1945cd6c19e…`. The titles say *Byte-Exact* and *Capsule*; the bodies are empty. **A
document titling itself byte-exact does not make it so**, and the `EMPTY_NATIVE_BODY` label
the 2026-09-17 inventory already carried for all four is the accurate one.

## The controlling status banners, verbatim

### The two registers this lane sits under

`registers/json/artifact_index.json`, row `P1.1`, records the status of the priority brief
that defines this lane as

> NONCANONICAL q0-law-specific/1.2 FAIL-CLOSED PASS / 12-OF-12 NEGATIVE TESTS / INDEPENDENT REVIEW OPEN

and its canonical impact as

> No canonical impact; active q0_machine.json and q0_verify.py remain authoritative and untouched.

`registers/json/alarms.json` carries a CRITICAL alarm `Q_MACHINE_PROTOTYPE_REPAIR_REQUIRED`
against `LAW_SPECIFIC_Q_SUCCESSOR_MACHINE` under transition `TR-QM-001`:

> AO48 hand trace found reversed typed-rate wiring, missing R0, and verifier gaps. GP-REG-021/GP-DATA-022 must not be promoted or installed.

and a WARNING alarm `Q_MACHINE_V11_INDEPENDENT_EXECUTION_PENDING` under review route
`REV-QM11-GP-001`:

> The corrected successor is author-built and locally executed. It must not be installed until an outside line independently executes the exact source and confirms theorem-graph reachability.

### The primary machine root

`q0_verify.py` opens

> q0_verify.py — CONSOLIDATED EXECUTABLE VERIFICATION LAYER (C-CONS-2026-07-19)

and `q0_machine.json` states its own consolidation policy as

> Append-only discipline preserved. C092/C093 archives remain immutable at their recorded hashes; this set is a successor consolidation, not an edit.

Neither was executed here, and nothing in this repository reads either of them.

### The audit that opened the alarm

`AO48-AUD-008 - Successor machine audit …` carries the header block

> ARTIFACT ID: AO48-AUD-008-v1.0

> STATUS: PROPOSED; §4's trace results are CONFIRMED-BY-HAND-TRACE, not execution — I have no runtime and say so

> AUTHORITY: none

> CANONICAL IMPACT: NONE — and the active q0_machine.json is untouched by this audit as by the proposals it audits

Its headline finding is stated as

> **`Q_TYPED_MS_RATE` is wired backwards.**

### The registry the migration classifies

`q0_registry_v2_0.json` records, in its own metadata, the classification of the legacy
registry it migrated:

> LEGACY SCHEMA DEMONSTRATION. It is not the frozen Q0 Rate Program theorem registry and its constants are not Q0 theorem inputs.

and the reason the identifier `R0` must not be read across the two:

> Legacy ID R0 denoted a remainder condition, not the canonical Gaussian-Sard/Morse-Smale R0 of the Q0 program.

### What the cycle contracts say about themselves

`q0_c101_limit_contract.json` records its own status as `CORE-CLOSED-INTERNAL-PROGRAM-GRADE`
with `R0_external_status` of `SPECIALIST-REVIEW-PENDING` and both
`numeric_upper_coefficient` and `numeric_lower_coefficient` set to `NOT-CLAIMED`.
`C101_RELEASE_README.md` puts the same boundary in prose:

> No numerical value of \(C_{Q0}\) is claimed.

`q0_c097_gamma_contract.json` records its status as

> PROVEN-MODULO seven explicit obligations

and `q0_c108_portfolio_contract.json` its terminal state as `RELEASE-CLOSED` with the
external dependencies

> independent SARD-G specialist acceptance

Read the per-directory READMEs for the rest; each quotes the banners of the objects it
holds.

## Sub-directories

| directory | items | README |
|---|---:|---|
| `02.1_MACHINE_ROOT_SCHEMA/00_NAVIGATION_AND_STATUS` | 2 | [README](02.1_MACHINE_ROOT_SCHEMA/00_NAVIGATION_AND_STATUS/README.md) |
| `02.1_MACHINE_ROOT_SCHEMA/01_PRIMARY_MACHINE_ROOT` | 2 | [README](02.1_MACHINE_ROOT_SCHEMA/01_PRIMARY_MACHINE_ROOT/README.md) |
| `02.1_MACHINE_ROOT_SCHEMA/02_VERIFIER_AND_KERNEL_LINEAGES` | 7 | [README](02.1_MACHINE_ROOT_SCHEMA/02_VERIFIER_AND_KERNEL_LINEAGES/README.md) |
| `02.1_MACHINE_ROOT_SCHEMA/03_REGISTRIES_AND_STATES` | 6 | [README](02.1_MACHINE_ROOT_SCHEMA/03_REGISTRIES_AND_STATES/README.md) |
| `02.1_MACHINE_ROOT_SCHEMA/04_CYCLE_AND_PORTFOLIO_CONTRACTS` | 9 | [README](02.1_MACHINE_ROOT_SCHEMA/04_CYCLE_AND_PORTFOLIO_CONTRACTS/README.md) |
| `02.1_MACHINE_ROOT_SCHEMA/05_SECOND_DOMAIN_CONTRACTS` | 1 | [README](02.1_MACHINE_ROOT_SCHEMA/05_SECOND_DOMAIN_CONTRACTS/README.md) |
| `02.2_NEGATIVE_TEST_BATTERIES/00_NAVIGATION_AND_SCOPE` | 1 | [README](02.2_NEGATIVE_TEST_BATTERIES/00_NAVIGATION_AND_SCOPE/README.md) |
| `02.2_NEGATIVE_TEST_BATTERIES/01_CANONICAL_TEST_RECEIPTS` | 1 | [README](02.2_NEGATIVE_TEST_BATTERIES/01_CANONICAL_TEST_RECEIPTS/README.md) |
| `02.2_NEGATIVE_TEST_BATTERIES/02_ADVERSARIAL_HARNESS_AND_REEXECUTIONS` | 2 | [README](02.2_NEGATIVE_TEST_BATTERIES/02_ADVERSARIAL_HARNESS_AND_REEXECUTIONS/README.md) |
| `02.2_NEGATIVE_TEST_BATTERIES/03_REDTEAM_SOURCE_AND_RESULTS` | 2 | [README](02.2_NEGATIVE_TEST_BATTERIES/03_REDTEAM_SOURCE_AND_RESULTS/README.md) |
| `02.3_VALIDATION_RECEIPTS/00_NAVIGATION_AND_SCOPE` | 1 | [README](02.3_VALIDATION_RECEIPTS/00_NAVIGATION_AND_SCOPE/README.md) |
| `02.3_VALIDATION_RECEIPTS/01_VALIDATION_SUMMARIES_AND_RUN_BLOCKERS` | 5 | [README](02.3_VALIDATION_RECEIPTS/01_VALIDATION_SUMMARIES_AND_RUN_BLOCKERS/README.md) |
| `02.3_VALIDATION_RECEIPTS/02_RAW_VALIDATION_LEDGERS` | 3 | [README](02.3_VALIDATION_RECEIPTS/02_RAW_VALIDATION_LEDGERS/README.md) |
| `02.3_VALIDATION_RECEIPTS/03_SOURCE_CAPSULE_AND_PAIRING_INSTRUMENT_AUDITS` | 6 | [README](02.3_VALIDATION_RECEIPTS/03_SOURCE_CAPSULE_AND_PAIRING_INSTRUMENT_AUDITS/README.md) |
| `02.3_VALIDATION_RECEIPTS/04_VALIDATION_SCRIPTS_AND_BATTERIES` | 9 | [README](02.3_VALIDATION_RECEIPTS/04_VALIDATION_SCRIPTS_AND_BATTERIES/README.md) |
| `02.3_VALIDATION_RECEIPTS/05_CONTRACT_REPORTS_AND_VALIDATION_OUTPUTS` | 18 | [README](02.3_VALIDATION_RECEIPTS/05_CONTRACT_REPORTS_AND_VALIDATION_OUTPUTS/README.md) |
| `…/06_…/00_RELEASE_AUDITS` | 5 | [README](02.3_VALIDATION_RECEIPTS/06_RELEASE_AUDITS_ATTESTATIONS_AND_EXTRACTION_TESTS/00_RELEASE_AUDITS/README.md) |
| `…/06_…/01_FRESH_EXTRACTION_TESTS` | 5 | [README](02.3_VALIDATION_RECEIPTS/06_RELEASE_AUDITS_ATTESTATIONS_AND_EXTRACTION_TESTS/01_FRESH_EXTRACTION_TESTS/README.md) |
| `…/06_…/02_ATTESTATIONS_AND_RELEASE_READMES` | 6 | [README](02.3_VALIDATION_RECEIPTS/06_RELEASE_AUDITS_ATTESTATIONS_AND_EXTRACTION_TESTS/02_ATTESTATIONS_AND_RELEASE_READMES/README.md) |
| `…/07_…/00_DEEP_REVIEWS_AND_INDEPENDENT_CHECKS` | 8 | [README](02.3_VALIDATION_RECEIPTS/07_ADJUDICATIONS_REVIEWS_AND_SOURCE_AUDITS/00_DEEP_REVIEWS_AND_INDEPENDENT_CHECKS/README.md) |
| `…/07_…/01_ADJUDICATIONS_AND_SUPERSESSIONS` | 5 | [README](02.3_VALIDATION_RECEIPTS/07_ADJUDICATIONS_REVIEWS_AND_SOURCE_AUDITS/01_ADJUDICATIONS_AND_SUPERSESSIONS/README.md) |
| `…/07_…/02_SOURCE_AUDITS_AND_REPRODUCTION_NOTES` | 2 | [README](02.3_VALIDATION_RECEIPTS/07_ADJUDICATIONS_REVIEWS_AND_SOURCE_AUDITS/02_SOURCE_AUDITS_AND_REPRODUCTION_NOTES/README.md) |
| `…/07_…/03_TRANSFER_CERTIFICATES_AND_TECHNICAL_FINDINGS` | 6 | [README](02.3_VALIDATION_RECEIPTS/07_ADJUDICATIONS_REVIEWS_AND_SOURCE_AUDITS/03_TRANSFER_CERTIFICATES_AND_TECHNICAL_FINDINGS/README.md) |

The four items at this root are the LS-DATA capsule documents tabled above; their rows are
in the root `_MANIFEST.jsonl`, and the 29 folder rows are indexed in that same manifest, so
the lane's item count balances at one row per Drive id.

## What was deliberately not ported

Nothing. Every one of the lane's 145 inventory items has a row, and every one of its 116
files has bytes on disk. The only items without bytes are the 29 folders, which have no
bytes to store.

Nothing was fetched from `99_DO_NOT_OPEN` (Drive id
`1VTiBaRlBvHGptiXqoEli4E5630nO7mNb`), which this lane does not touch in either direction.

## What this does not establish

This is the load-bearing section, and it is longer than the rest on purpose.

* **No status moved.** Nothing here promotes, closes, discharges, reclassifies or grades
  any claim, premise or obligation. Every status word above is transcribed from a source
  and attributed to it. A document in this lane calling itself a certificate, an
  attestation, a release audit or a closure does not make it one, and this repository does
  not adopt any of those words as its own.
* **A digest match is an identity of bytes and nothing more.** That the 91 byte-exact
  files hash to what the 2026-09-17 inventory declares says the repository holds the same
  bytes the accessibility publication catalogued. It says nothing about whether those bytes
  are correct, current, authoritative, or mathematically sound.
* **Nothing here was executed.** The lane holds 18 Python files, several of which describe
  themselves as verifiers, kernels, batteries or audits. None was run, none was imported,
  no test was reproduced, and no result reported inside any of them was checked. The PASS
  and FAIL rows inside the stored JSON are the source's own records of runs this repository
  did not perform.
* **The reading copies are not the objects.** Twenty-five native Google Docs are held only
  as text exports. Their content was read to write this file; their bytes were not, because
  a native Doc has no payload digest to check against. Quoting an export is quoting the
  export.
* **The three-byte capsules are not recoveries.** See above: `recovery/LEDGER.json` records
  those Drive ids as UNRECOVERABLE and that finding stands unchanged.
* **This lane is NONCANONICAL and its gates are open.** The register rows quoted above say
  so in the source's own words: distinct-family review and Dylan Roy's promotion are
  required, and independent execution by an outside line is pending. Mirroring the bytes
  supplies none of that. A same-provider reading of a same-provider artifact earns zero
  independence credit, and aging earns none either.
* **No original prize problem is solved**, and nothing in this lane may be composed with
  the 2D upper/lower tracks or the 3D lifetime track across the withdrawal recorded in
  `ERRATA_AND_CLARIFICATIONS_2026-09-13`.
* **A green checker run is not a certificate.** `tools/verify_manifests.py` proves the
  digests and byte counts in these manifests describe the files on disk;
  `tools/mirror_quotes_check.py` proves the quotation marks above enclose strings that are
  present in the stored bytes. Neither reads a sentence, checks an attribution, or has any
  opinion about the mathematics.
