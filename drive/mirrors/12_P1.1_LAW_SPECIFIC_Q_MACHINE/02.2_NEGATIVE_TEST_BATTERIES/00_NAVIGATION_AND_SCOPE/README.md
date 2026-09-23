# `02.2_NEGATIVE_TEST_BATTERIES/00_NAVIGATION_AND_SCOPE`

Drive folder id `1tVRfhFD8qTX45x0XkMgG4zlGKJygwlEt`. The 2026-09-17 inventory gives this
folder **1 item**, a native Google Doc, held here as a text export — a reading copy, not
the object.

## What is held, and at what exactness

| stored file | exactness | bytes on disk | Drive-reported size |
|---|---|---:|---:|
| `00_LEAF_CARD — 02.2 Negative Test Batteries (CL-NAV-036.7).export.txt` | reading copy | 2,590 | 2,974 |

## The status banners, verbatim

The card's own header disclaims authority:

> ARTIFACT: CL-NAV-036.7-v1.0 · AUTHOR: Claude (Anthropic), CL-* instance · CREATED: 2026-07-21 · CLASS: NAV — leaf charter + live-state card · CANONICAL IMPACT: NONE · AUTHORITY: none

Its design principle is stated as

> A negative test exists to catch a named defect class. Every defect found by audit becomes a mutation that the battery must kill; a battery that passes its author's artifact but has no mutation for a known defect class is coverage theater (cf. OP-GDN-002 §8 — serious errors become reusable controls).

and its execution discipline as

> Batteries run against frozen sources with recorded hashes; results are receipts (see 02.3), not prose claims. Author self-runs are LOW independence (REV-*-SELF pattern, CM-046) and must be labeled as such; independent second-stack execution is what upgrades them.

What it says must not be claimed:

> Passing a battery certifies only the covered defect classes. "All tests pass" without the mutation list is not evidence.

## What this does not establish

The card is a charter for work that lands in the sibling folders, not evidence about any of
it. Its own rule — that an author self-run earns LOW independence and that a pass certifies
only the covered defect classes — is the right frame for reading everything under
`02.2_NEGATIVE_TEST_BATTERIES` and `02.3_VALIDATION_RECEIPTS`, and this repository adopts
none of the passes it describes. The Doc's bytes are not held; only the export is.
