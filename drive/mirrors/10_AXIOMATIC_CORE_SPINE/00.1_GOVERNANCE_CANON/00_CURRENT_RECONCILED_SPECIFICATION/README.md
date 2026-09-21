# `00.1_GOVERNANCE_CANON/00_CURRENT_RECONCILED_SPECIFICATION`

Drive folder id `1_e7V938L5bKbni1jDI6DbPrpbj25bdAk`, **2 inventory items**: the
folder itself (indexed tree-only in the lane-root manifest) and one file, held
here byte-exact.

| file | identity |
|---|---|
| `GATE_FRAMEWORK_MASTER — CURRENT-v1.2-RECONCILED — 30769B — SHAea3b0bce — ID19vdTHxH.md` (30,769 B) | **byte-exact**: SHA-256 `ea3b0bce…` and byte count equal the `drive/inventory.jsonl` row for id `19vdTHxH0ZvRtT20R5U9DPeWLdHtH-vnJ` |

## The status banners, verbatim

The header records "Version:** Master v1.2 (reconciled)", "Date:** 2026-07-17",
"Status:** current successor specification", and under
"Supersedes without overwriting:" lists the Master v1.0 PDF, the uploaded Master
v1.1 Markdown, and the C095 executable-specification line also named v1.1. It
declares "Established machinery:** one mandatory SCHEMA preflight, 16 numbered
gates, five typed schema profiles, canonical dependency hashes, and explicit
conditional-debt propagation." It also carries a standalone notice at lines 18-21,
set as a bold blockquote, reproduced here with the source's own `>` prefixes and
emphasis:

> **Standalone notice.** This document contains the complete conceptual and
> machine-contract specification. It can be implemented without any earlier Q0
> file. Earlier versions remain indispensable provenance and retrodiction
> evidence, but they are not needed to understand or apply this version.

Read it as the document's claim about itself. That a specification says it can be
implemented without any earlier file is not a finding of this repository, and
"indispensable provenance and retrodiction evidence" is the source's grading of
its predecessors, not one applied here.

Its own limits are stated in §0.2 and §0.3:
"The machine certifies **tag consistency and arithmetic-contract consistency**.
It does not, by itself, certify tag-to-content fidelity."; and
"A gate PASS never means “the theorem is true.”" The division of labour it draws
from that is set, at lines 79-81, as a bold blockquote, reproduced here with the
source's own `>` prefixes and emphasis rather than stripped of them:

> **Gates classify and block structurally invalid reuse. Proofs,
> recomputation, adversarial reimplementation, source retrieval, and separated
> adjudication detect content error.**

Appendix B is a list of statements the file forbids, opening "Do not say:" and
including "“a gate PASS proves the theorem”", "“the kernel checks mathematical
truth”", "“two agreeing implementations certify correctness” without
COMMON-MODE", "“a measured coefficient is scale free”" and "“a theorem is
uniform” from sampled rungs".

## What this does not establish

The byte-exact row establishes that this directory holds the same bytes the
2026-09-17 inventory declares for that Drive id, and nothing further. The file
calls itself the current successor specification; so does one of the two files
in the sibling `01_PRIOR_DIVERGENT_V1_1_LINEAGES`, with a different gate count,
and this repository does not choose between them. Mirroring is not review,
replay, endorsement or promotion; the gate names, verdict words and machinery
counts above are the source's own and none was decided here. No gate described
in this file is run by anything in this repository, and nothing here grades,
closes or discharges any claim, premise or obligation.
