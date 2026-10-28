# Recover accidental Library removals

The intended reader journey is source → saved context → reference → resume. On exact Main a15d2fb an independently reproduced Remove/Clear permanently replaces the local list, including in peer tabs. The delegated owner request authorizes this small repair without another design-approval round.

## Approach

Use a single tab-local recovery buffer for the latest successful Remove or Clear. Reuse the existing locked mutation queue and `mergeEntries` validation. A confirmation dialog would interrupt intentional work and still offer no post-action recovery; a persisted multi-step history adds a second storage/concurrency protocol without a demonstrated need. One-step additive Undo is the chosen implementation judgment.

Capture only entries actually removed from the current base **inside** the lock, not an old whole-list snapshot. A no-op removal does not overwrite recovery. Undo reads the latest base and merges removed entries, preserving peer additions and the established source-metadata conflict rules, 50-entry and128KiB ceilings. Refusal leaves the list unchanged and recovery available. A successful Undo consumes recovery. Further saves/imports/storage events do not discard it; a later actual Remove/Clear replaces it. Recovery lasts only in this page instance and is not persisted; a new page load starts without it. No timer or cross-tab recovery claim.

Add one native “Undo removal” button beside existing list tools, hidden/disabled until available, and a brief explanatory sentence. Announce availability/refusal/completion through the existing status text. Disabling/hiding a focused Undo returns focus to the reading-list summary; queued completion does not steal a reader's newer outside focus. No layout redesign or dependency.

## Boundaries

No new schema, storage key, network/service, source acquisition, mathematical record, status, certification, Atlas or importer changes. Preserve explicit tab-only/storage refusal behavior. Existing Drive source a514 (independently downloaded and verified) exactly matches all four planned source preimages, verified before edits in PREIMAGES.json. No original is deleted.

## Acceptance

Reproduce missing Undo on unmodified source. Real browser checks cover source+search exact metadata restoration, keyboard removal/clear/Undo, two-tab additive preservation, deduplication, current metadata conflict,50-entry/128KiB capacity refusal and retry, most-recent/no-op semantics, queued operations, tab-only and storage/lock refusal, reload lifetime, visible focus/overflow at1440/390/320 light/dark. Re-run existing reading continuity/list/Cite checks plus prescribed public-shop commands. Separate eligible engineering review and distinct nonauthor/nonreviewer protected integration precede release. Evidence establishes only bounded engineering behavior.
