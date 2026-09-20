# `00.1_GOVERNANCE_CANON/03_OPERATOR_GUIDANCE_AND_RECOVERY`

Drive folder id `1p999Kg9MlbpEqLjvxUcDw68wiLOh3j3u`, **4 inventory items**: the
folder itself (indexed tree-only in the lane-root manifest), one `text/markdown`
file held byte-exact, and two native Google Docs held as reading copies.

| file | identity |
|---|---|
| `AO48-OPR-022 - Operator guidance relayed verbatim - Coupled Mathematical-Architectural Advancement and Required Safeguards` (14,914 B) | **byte-exact**: SHA-256 `236f0077…`, id `1enUo1aKmuK2RMvEmixGc3x8c2Ug-veku`. The Drive title carries no file extension; the stored name is the title |
| `OP-GDN-002 — Coupled Mathematical–Architectural Advancement and Required Safeguards.export.txt` (7,546 B export) | **reading copy**, `exact: false`, id `1-9ePRpp28DN19YFbdsGqKVXfIAdnScyt8BpV0lEdY8A`; no payload digest for it exists in the corpus |
| `OP-DIR RECOVERY — Model Coordination and Approval Protocols (original + corrected), transcribed from root mobilebasic captures (CL-REC-038).export.txt` (4,947 B export) | **reading copy**, `exact: false`, id `1PjpcjsIk3SsNo8864FoLLxCvTjxI7grd_IWjvPn2QLg` |

## The status banners, verbatim

**`AO48-OPR-022`** is a relay, and says so in its own header block:
"STATUS:            RELAY — the text below the rule is the operator's,
unaltered."; "AUTHORITY:         operator (relayed); none added by the relaying
line"; "CANONICAL IMPACT:  NONE by the relay itself; the guidance's own force is
the operator's to assert". The header adds that if a canonical OP-GDN version
exists or appears through another channel, "that version governs and this relay
corroborates it." Two of the relayed guidance's own fences read
"Silence does not count as evidence that the architecture kept pace." and
"Architecture must not grow merely because additional structure is possible."

**`OP-GDN-002`** is that canonical version, and carries
"Canonical impact: Governance only. Mathematical truth remains evidence-governed."
Its §5 is headed "CONSENSUS-LAUNDERING CONTROL" and its §3 "SYNCHRONIZED-CALIBRATION
RISK", but the sentence "This is the synchronized-calibration risk." is not in
OP-GDN-002 at all: the only stored byte in this lane that carries it is the
`AO48-OPR-022` relay below, at its line 103.

**`OP-DIR RECOVERY` (CL-REC-038-v1.0)** is a transcription wrapper, and separates
its own authority from the text it carries:
"CANONICAL IMPACT: NONE · AUTHORITY: none (the recovered text is
operator-authored; this wrapper is not)". It states its method:
"Transcription method: HTML tag-stripping of the captures; text below is
verbatim except whitespace normalization. CL did not alter, reorder, or
summarize the directive text." Its closing section is explicitly marked as the
wrapper's own reading and not part of the directive.

## What this does not establish

The byte-exact row establishes that this directory holds the same bytes the
2026-09-17 inventory declares for `AO48-OPR-022`, and nothing further. The two
reading copies are not the objects: a text export of a native Google Doc loses
the object's own bytes and carries no declared digest, so their manifest digests
describe the exports and nothing else.

Both `AO48-OPR-022` and `OP-DIR RECOVERY` describe themselves as carrying
operator text verbatim. **This repository has not verified either claim**: the
relay's upstream chat line and the recovery's six root-level HTML captures are
not held here, and neither the inventory nor any register in this repository
binds them to these bodies. What is established is that the bytes of the relay
match the inventory digest for its id — identity of bytes, not fidelity to a
source those bytes describe.

Mirroring is not review, replay, endorsement or promotion. The operator
authority, approval and governance words above are the sources' own; none was
decided, exercised or graded here, and nothing here closes or discharges any
claim, premise or obligation.
