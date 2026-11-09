# QP-C54 errata of 9 November 2026: procedural review out of `blocking_reason`

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE.

## Why

OpenAI Codex's bounded field check of tranche 10 (main#684 comment 6748878712, 19:07Z) found review language in five blocking
reasons. Under the owner's directive a missing procedural review is not a mathematical blocking reason, so such notes must be kept
apart from any mathematical obstruction. I then scanned the `blocking_reason` of **every** QP-C54 record (tranches 1–10) for review
language: 22 sentences in 19 records.

## Items (`errata.json`, 11 items, 7 records)

Retained tranche files are never edited. Each item sets one field of one record, and `previous_value_sha256` pins the text it
replaces. A builder applies the items field by field and tags each row `errata_applied`.

| Record | Change | Source |
|---|---|---|
| QP-C54-T10-019 | review-status sentence moved to `procedural_notes` | Codex |
| QP-C54-T10-020 | "no independent review … recorded" moved to `procedural_notes` | Codex |
| QP-C54-T10-049 | "no independent review covers its translation" restated as the alignment gap it is: the SMT encoding is not shown faithful to PROOF.md | Codex |
| QP-C54-T10-047 | "has not been reviewed" restated as an alignment gap | scan |
| QP-C54-T2-024 | "not reviewed in the object" restated as "not established in the object" | scan |
| QP-C54-T3-038 | "Slice A/B analytic reviews … pending" moved to `procedural_notes` | scan |
| QP-C54-T5-003 | the paragraph on the missing non-Claude delta alignment review moved to `procedural_notes` | scan |

Every record keeps a mathematical blocker: no indexed kernel receipt, inputs consumed but not proved, an alignment gap, or
statements with no Lean form.

## Left unchanged, by design

- **QP-C54-T10-008 and -009.** "Review" there refers to the source recording a hypothesis defect, which is mathematical content.
  Codex reached the same reading.
- **The other 13 scan sentences** (T2-035, T3-004, T3-006, T3-008, T3-030, T3-035, T4-008, T4-026, T4-079, T4-101, T6-004,
  T8-018). They cite a source's review document while stating a substantive gap: unwritten steps, imported or unproved inputs,
  sketches, derivations not run, or programs not executed.

## Limits

- The scan is a literal word search, by an Anthropic Claude session, and judgment decided each case.
- Whether any remaining blocker is the right mathematical obstruction is not reviewed here.
