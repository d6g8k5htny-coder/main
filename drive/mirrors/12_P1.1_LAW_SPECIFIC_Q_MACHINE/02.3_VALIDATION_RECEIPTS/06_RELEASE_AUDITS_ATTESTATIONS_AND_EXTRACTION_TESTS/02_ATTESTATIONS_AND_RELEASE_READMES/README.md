# `…/06_RELEASE_AUDITS_ATTESTATIONS_AND_EXTRACTION_TESTS/02_ATTESTATIONS_AND_RELEASE_READMES`

Drive folder id `10p4cnbkEvmTNvr89hZofjEed7XknrJgo`. The 2026-09-17 inventory gives this
folder **6 items** — 4 `application/json` and 2 `text/markdown` — and all 6 are held
byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `C095_ATTESTATION.json` | byte-exact | 2,025 |
| `C096_ATTESTATION.json` | byte-exact | 2,991 |
| `C096_REPRODUCTION_README.md` | byte-exact | 5,179 |
| `C101_ATTESTATION.json` | byte-exact | 2,322 |
| `C101_RELEASE_README.md` | byte-exact | 2,339 |
| `C108_ATTESTATION.json` | byte-exact | 1,521 |

## The status banners, verbatim

`C095_ATTESTATION.json` records the current Q0 status of its cycle as

> "structural_UBG": "DERIVED",

> "uniform_4p35": "BLOCKED on UB_G_RESIDUAL_UNIFORM",

> "q0_via_cubic_rate": "PROVEN-MODULO UB_G_RESIDUAL_UNIFORM",

`C096_ATTESTATION.json` records

> "unconditional_promotable": false

and its principal findings include

> "uniform_4p35": "blocked",

> "q0_limit": "Proven-Modulo three explicit residual conditions",

> "legacy_registry": "archive-valid migration; not candidate or promotion valid",

`C101_ATTESTATION.json` records the terminal status

> "terminal_status": "CORE-CLOSED-INTERNAL-PROGRAM-GRADE",

and a `not_claimed` list that reads, in full,

> "No numerical value of C_Q0.",

> "No restoration of 4.3 or 4.35 upper displays.",

> "No finite numerical lower coefficient.",

> "No thermodynamic-limit or critical-height theorem.",

> "No external specialist acceptance of SARD-G."

`C108_ATTESTATION.json` records the portfolio's terminal state and its per-project states:

> "Q0-SHARP": "KILLED",

> "Q0-LLM": "BLOCKED-EXTERNAL"

and its summary statement:

> The Q0 successor portfolio is byte-verified, fresh-extraction verified, semantically reconciled, and terminally release-closed. All remaining dependencies are explicitly external or require a new successor freeze.

`C096_REPRODUCTION_README.md` states what its release does and does not claim:

> ## What is not claimed

> - A gate PASS does not prove mathematical truth.

> - The three residual conditions in the Q0 contract are not discharged here.

> - The decimal upper coefficient 4.35 is not promoted.

> - The Q0 selection limit is not claimed unconditionally by this release.

> - The BR-MARK theorem is not closed.

`C101_RELEASE_README.md` states its theorem boundary and then

> No numerical value of \(C_{Q0}\) is claimed.

with its own "Not claimed" list, including

> - No external specialist acceptance of SARD-G.

> - No publication-grade interval certificate for the existence constants.

## What this does not establish

An attestation is a document that attests. It is not a certificate, and this repository does
not adopt one word of it. Read the two release READMEs' own "not claimed" lists: the sources
themselves say that a gate PASS is not truth, that the residual conditions are not
discharged, that no decimal coefficient is promoted, and that no external specialist has
accepted SARD-G. Those lists are the most reliable content in this folder, and nothing in
this port weakens them. Nothing here was reproduced, no bundle hash was recomputed, and no
status moved.
