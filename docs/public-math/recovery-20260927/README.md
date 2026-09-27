# Source reconciliation — 27 September 2026

**Scope: source custody, discovery, and one isolated calculation helper. Scientific effect: NONE.**
This dated audit supplements the [existing public source catalog](../sources.json); it is not a replacement catalog or a scientific-status register.

## Fresh result

| Measure | Before this continuation | After fresh byte checks |
|---|---:|---:|
| R2 unmatched rows | 442 | 299 |
| R2 unmatched distinct hashes | 428 | 286 |
| Outstanding execution-manifest hash leads | 142 | 0 |

All **142 outstanding manifest-only hashes** now match actual Drive downloads, resolving **143 R2 rows** because one hash occurs twice. In total this continuation checked **147 research-file downloads / 146 distinct hashes / 10,363,392 bytes**. Every expected size and SHA-256 matched. The first audit's 24-file evidence manifest was also reverified.

This does not recover every input of the old execution receipts or reconstruct their dirty working trees. Their reported 2,890-test suites were not rerun. It does not establish fresh absence or presence across every GitHub branch.

The [Drive audit folder](https://drive.google.com/drive/folders/1fyBQIGagU-iACWnTqKYpX8-fHmUc6oIr) is the destination for the detailed report, per-file verification ledger, remaining-R2 ledger, metadata-only leads, workbook, and bounded replay evidence. Drive records retain their existing access boundaries; this is not a claim that all raw research files are newly public.

## C027: already in Drive, not missing bytes

- [Original station program](https://drive.google.com/file/d/15FVUpBdsTZWotDmAJI8K5V3BAi2Sm5Um/view): 5,415 bytes; SHA-256 `ec6ee2f49b9e4f54b07a647bd14371d84c1c0dfa78296eba82e2e222c0fc1d3d`.
- [Original coarse table](https://drive.google.com/file/d/142nDsWYKZm5pauaYt4cTpLY_dSrjb1b6/view): 44,087 bytes; SHA-256 `d6f1f9e61a0c2d4ee70cfd1ed7eb4e13fb5f9dba9aa76109677bcdc2aacb0abd`.
- [C026 dependency](https://drive.google.com/file/d/1zl8NNCsaAa8tgFwhMf2-s1zwY0cOk8Z0/view): 6,553 bytes; SHA-256 `4058cad4cc7c69201cb6f7ea392c1fae1099d087a7489a34e01ece064282519a`.

The existing `c027_station.py` variant changes only an absolute import path to `.`. A byte-identical dependency with the importable underscore filename also already exists. No duplicate canonical Drive sources were created and no original was overwritten.

At scaled station `(-3,1)`, the unchanged original program reproduced all eight stored fields in ordinary and optimized Python 3.13.5. Only **one of 258 table rows** was replayed. This is not a rigorous intensity enclosure, a full-sweep reproduction, or a theorem certificate.

## Reproduced bug and separate successor

The dependency's `LS.inv()` returns linear coefficient 0 rather than -1 for `1/(1+r)`. The isolated regression fails in both interpreter modes. No calls to that helper occur in the inspected C026/C027 pair; wider callers have not been audited.

[Math PR91](https://github.com/d6g8k5htny-coder/Math-/pull/91) adds a separate exact **finite-polynomial** reciprocal helper, derivation, tests, and CI. It merged at `3b2ac59f8f02b9573d01087aefff85bbc279a55a` after both hosted workflows succeeded. [Read the pinned scope and derivation](https://github.com/d6g8k5htny-coder/Math-/blob/3b2ac59f8f02b9573d01087aefff85bbc279a55a/repairs/c026_reciprocal_20260927/README.md).

Twelve unit tests pass in each Python mode; two deliberate algorithm mutations are rejected in both modes. The helper is **not installed into the historical engine** and does not silently treat unknown-tail truncated series as complete finite polynomials. The entire pre-existing Math suite was not run locally. No nonauthor acceptance is claimed; the automated Codex review reported a usage limit, not a review verdict.

## Discovery correction and next queue

A selected-extension catalog or MIME-filtered search miss is **not evidence of source absence**. Use the [existing all-type Drive source map](https://docs.google.com/spreadsheets/d/1hO3MPQwtAiEjCGdOjZJTbB8fKIvt_3GU6KgxPGTPiLg/edit), or enumerate the known parent folder, then retrieve bytes and compare size/SHA-256. Distinguish current default-branch paths from immutable historical paths. A native-Doc reading export is not automatically the original file bytes.

Of the **286 remaining R2 hashes**, **242 have metadata locators awaiting byte checks** and **44 have no locator in this particular source map**. Neither class proves global absence. `TRANSVERSE_CONTACT_ASYMPTOTIC.md` remains unrecovered. No Vault99 original was opened.

## Coordination and authority

The [existing owner-delegation record](../../../governance/OP-AUTONOMY-20260923-v2.1.md) now includes Dylan's September 27 reaffirmation: act autonomously in line with project intent, coordinate conflicting rules, and implement justified corrections. This does not change credentials or platform controls. A rejected direct-main write changed nothing; this change follows the required PR route.

Work was coordinated in [main issue86](https://github.com/d6g8k5htny-coder/main/issues/86#issuecomment-5858217205), with the [bounded reciprocal scope update](https://github.com/d6g8k5htny-coder/main/issues/86#issuecomment-5858478286). Requests to other agents are not recorded as completed reviews. No proof body, acceptance record, premise, prize, or `lemma_closed` value changed.
