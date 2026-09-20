# `02.2_NEGATIVE_TEST_BATTERIES/01_CANONICAL_TEST_RECEIPTS`

Drive folder id `1yZpYjIlN0E0Oi7QXvtnetHn8bTGkuamg`. The 2026-09-17 inventory gives this
folder **1 item**, a native Google Doc, held here as a text export — a reading copy, not
the object. The export carries a Python source listing inside it; that listing is text
inside a Doc, not a stored `.py` object, and it has no digest in the corpus.

## What is held, and at what exactness

| stored file | exactness | bytes on disk | Drive-reported size |
|---|---|---:|---:|
| `GP-DATA-052-v1.0 — Law-specific q verifier and 16-test receipt.export.txt` | reading copy | 14,131 | 7,301 |

## The status banners, verbatim

The header block reads

> ARTIFACT ID:       GP-DATA-052-v1.0

> CLASS:             DATA — proposed verifier source and adversarial receipt

> AUTHORITY:         none

> CANONICAL IMPACT:  NONE

> STATUS:            LOCAL EXECUTION PASS / 16 OF 16 NEGATIVE TESTS PASS / NONCANONICAL

The receipt records its own verifier and machine digests, and its base-machine result
carries

> "negative_tests_passed": 16,

> "unconditional_promotable": false

for every root it reports.

## What this does not establish

The sixteen negative tests were run by the author, elsewhere, at a time the document
records. **Nothing was run here.** The document's own status line already says what the
result is worth — a local execution pass, noncanonical — and `registers/json/alarms.json`
records the standing WARNING that the corrected successor

> must not be installed until an outside line independently executes the exact source and confirms theorem-graph reachability.

Every root in the receipt is reported as not unconditionally promotable, by the receipt
itself. Holding the export establishes that this repository has read the claim, not that
the claim is true, reproducible, or independent.
