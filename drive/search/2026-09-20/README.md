# Derived research search inputs

This directory supplements the existing immutable migration and reconciliation
records. It does not replace the Drive governing register or create another
manually maintained catalog.

`deliveries.json` identifies six raw-readback-verified outputs made after the
reconciled membership cutoff. Their exact bytes are retained under `deliveries/`.
Their titles were checked against Drive metadata in this pass. Existing
inventory, scope, coverage and archive tables remain byte-for-byte unchanged.

`text-coverage.json` pins the complete deterministic extraction of 3,140 eligible
stored files, including per-record disposition. Text is regenerated into a
disposable local cache; all source chunks and raw bytes are verified before
extraction, and each query checks the complete decompressed cache digest.
No held, personal, vault, quarantine, legacy or history content is opened.
Archive members remain metadata-only in this text search.

`navigator-native.xlsx` is the final export readback of the additive native
[Drive navigator](https://docs.google.com/spreadsheets/d/1x9aKuv9pQGD6kwwa3ZmehSyEnVX2ZlhhcY2OehjMCqI/edit).
`navigator-verification.json` records complete value and explicit-link checks.
This export is not represented as a revision-pinned native backup.

The current view combines 4,928 reconciled objects with six later deliveries,
plus 11,649 separately identified archive occurrences. Its own new navigator,
proof, bundle and final receipts are outside that input snapshot and are
identified in this pass's delivery manifest. No recursive re-ingestion is
required for completion. See `docs/DRIVE_SEARCH_GUIDE.md` for exact limits,
source roles, digest types and the active-package audit.

No scientific authority, theorem promotion, account-wide completeness or
independence credit is established.
