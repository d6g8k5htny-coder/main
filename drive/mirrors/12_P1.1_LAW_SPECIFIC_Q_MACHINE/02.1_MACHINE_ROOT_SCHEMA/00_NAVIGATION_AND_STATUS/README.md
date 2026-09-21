# `02.1_MACHINE_ROOT_SCHEMA/00_NAVIGATION_AND_STATUS`

Drive folder id `16ijjPpM3U4SU4Lf4WwkThlo1lk02rBCM`. The 2026-09-17 inventory gives this
folder **2 items**, both native Google Docs, and both are held here as text exports —
reading copies, not the objects. No payload digest for a native Doc exists anywhere in the
corpus, so neither row is `exact: true`.

## What is held, and at what exactness

| stored file | exactness | bytes on disk | Drive-reported size |
|---|---|---:|---:|
| `00_LEAF_CARD — 02.1 Machine Root Schema (CL-NAV-036.6).export.txt` | reading copy | 2,335 | 2,832 |
| `GP-REG-051-v1.0 — Law-specific q successor repair snapshot.export.txt` | reading copy | 13,316 | 6,192 |

The Drive-reported size and the export size differ because they measure different things:
the first is what Drive says the Doc weighs, the second is what the plain-text export
weighs. Neither is a digest.

## The status banners, verbatim

The leaf card carries its own non-authority header:

> ARTIFACT: CL-NAV-036.6-v1.0 · AUTHOR: Claude (Anthropic), CL-* instance · CREATED: 2026-07-21 · CLASS: NAV — leaf charter + live-state card · CANONICAL IMPACT: NONE · AUTHORITY: none

Its live-state entry on the prototype reads

> Prototype quarantined — REPAIR-REQUIRED: AO48-AUD-008 found by hand trace: (1) typed-rate dependency wiring in the wrong direction; (2) a required R0 node absent; (3) verifier holes not covered by the reported eight negative tests. The v1.0 prototype must not be promoted, installed, or described as qualified.

and the boundary it draws around the active machine is

> Hard boundary: the active q0_machine.json and q0_verify.py remain untouched. A corrected successor must add mutations for each identified defect (see 02.2 leaf) and receive independent execution before any migration proposal.

The repair snapshot beside it opens with its own supersession notice:

> SUPERSEDED FOR SUCCESSOR DEVELOPMENT — GP-AUD-188-v1.0 — 2026-07-24

> This q0-law-specific/1.1 snapshot is preserved as prior design provenance but is not the active repair candidate.

and its header block records

> STATUS:            PROPOSED / NONCANONICAL / VALIDATED LOCALLY / NOT APPROVED FOR INSTALLATION

with

> AUTHORITY:         none — asserts nothing on its own authority

> CANONICAL IMPACT:  NONE

## What this does not establish

The two documents here are navigation and provenance, not evidence. Holding their export
text establishes that this repository has read what the Drive shows and nothing more. The
quarantine, the supersession and the "NOT APPROVED FOR INSTALLATION" line are the sources'
own words, transcribed; this repository does not decide, confirm or lift any of them.
Neither Doc's bytes are held, so nothing here can be hashed against a Drive digest.
